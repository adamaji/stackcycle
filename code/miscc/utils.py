import os
import errno
import numpy as np

import pdb
from copy import deepcopy
from miscc.config import cfg

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils

import re, inspect
from torch import optim

#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def compute_discriminator_loss_cycle(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # # real pairs
    # inputs = (real_features, cond)
    # real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    # errD_real = criterion(real_logits, real_labels)
    # # wrong pairs
    # inputs = (real_features[:(batch_size-1)], cond[1:])
    # wrong_logits = \
    #     nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    # errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # # fake pairs
    # inputs = (fake_features, cond)
    # fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    # errD_fake = criterion(fake_logits, fake_labels)

    # if netD.get_uncond_logits is not None:
    real_logits = \
        nn.parallel.data_parallel(netD.get_uncond_logits,
                                  (real_features), gpus)
    fake_logits = \
        nn.parallel.data_parallel(netD.get_uncond_logits,
                                  (fake_features), gpus)
    uncond_errD_real = criterion(real_logits, real_labels)
    uncond_errD_fake = criterion(fake_logits, fake_labels)
    #
    # errD = ((errD_real + uncond_errD_real) / 2. +
    #         (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
    # errD_real = (errD_real + uncond_errD_real) / 2.
    # errD_fake = (errD_fake + uncond_errD_fake) / 2.
    
    errD_real = uncond_errD_real
    errD_fake = uncond_errD_fake
    errD = (errD_fake + errD_real) / 2.
    
    # else:
    #     errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data[0], errD_fake.data[0]

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # real pairs
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]


def compute_generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    #cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    # fake pairs
    # inputs = (fake_features, cond)
    # fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    # errD_fake = criterion(fake_logits, real_labels)
    #if netD.get_uncond_logits is not None:
    
    errD_fake = 0
    fake_logits = \
        nn.parallel.data_parallel(netD.get_uncond_logits,
                                  (fake_features), gpus)
    uncond_errD_fake = criterion(fake_logits, real_labels)
    errD_fake += uncond_errD_fake
    
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, encoder, decoder, image_encoder, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    torch.save(
        encoder.state_dict(),
        '%s/encoder_epoch_%d.pth' % (model_dir, epoch))    
    torch.save(
        decoder.state_dict(),
        '%s/decoder_epoch_%d.pth' % (model_dir, epoch))    
    torch.save(
        image_encoder.state_dict(),
        '%s/image_encoder_epoch_%d.pth' % (model_dir, epoch))
    
    print('Save all models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
            

from .dictionary import Dictionary
            
def load_external_embeddings(path):
    """
    Reload pretrained embeddings from a text file.
    """
    #assert type(source) is bool
    word2id = {}
    vectors = []

    # load pretrained embeddings
    #lang = params.src_lang if source else params.tgt_lang
    #emb_path = params.src_emb if source else params.tgt_emb
    #_emb_dim_file = params.emb_dim
    
    emb_path = path
    _emb_dim_file = 300
    
    with open(emb_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                assert word not in word2id
                assert vect.shape == (_emb_dim_file,), i
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            # if params.max_vocab > 0 and i >= params.max_vocab:
            #     break

    word2id["SOS_TOKEN"] = len(word2id)
    word2id["EOS_TOKEN"] = len(word2id)
    word2id["PAD_TOKEN"] = len(word2id)
    word2id["UNK_TOKEN"] = len(word2id)
    for i in range(4):
        temp = np.zeros((_emb_dim_file))
        temp[i] = 1
        vectors.append(temp[None])
            
    print("Loaded %i pre-trained word embeddings" % len(vectors))
    #logger.info("Loaded %i pre-trained word embeddings" % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id)
    dico.SOS_TOKEN = dico.word2id["SOS_TOKEN"]
    dico.EOS_TOKEN = dico.word2id["EOS_TOKEN"]
    dico.PAD_TOKEN = dico.word2id["PAD_TOKEN"]
    dico.UNK_TOKEN = dico.word2id["UNK_TOKEN"]
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    #embeddings = embeddings.cuda() if params.cuda else embeddings
    #assert embeddings.size() == (len(word2id), params.emb_dim), ((len(word2id), params.emb_dim, embeddings.size()))

    return dico, embeddings            

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
    elif method == 'sparseadam':
        optim_fn = optim.SparseAdam
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params