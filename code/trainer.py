from __future__ import print_function
from six.moves import range
from PIL import Image

import pdb

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
import os
import time

import numpy as np
import torchfile

from torchvision import models

import pickle

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model

from miscc.utils import KL_loss
from miscc.utils import compute_uncond_discriminator_loss, compute_cond_discriminator_loss
from miscc.utils import compute_image_gen_loss, compute_text_gen_loss
from miscc.utils import compute_latent_discriminator_loss, compute_latent_generator_loss
from miscc.utils import compute_uncond_generator_loss, compute_cond_generator_loss

from miscc.utils import load_external_embeddings, get_optimizer

from model import ImageGenerator, ImageEncoder
from model import TextGenerator, TextEncoder
from model import DiscriminatorLatent, DiscriminatorImage

from evaluation.evaluator import Evaluator

from visualizer import Visualizer

from tensorboardX import summary
from tensorboardX import FileWriter

class Trainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        # load fasttext embeddings (e.g., birds.en.vec)
        path = os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME + ".en.vec")
        txt_dico, _txt_emb = load_external_embeddings(path)
        txt_emb = nn.Embedding(len(txt_dico), 300, sparse=False)
        txt_emb.weight.data.copy_(_txt_emb)
        txt_emb.weight.requires_grad = False
        self.txt_dico = txt_dico
        self.txt_emb = txt_emb
        
        # load networks and evaluator
        self.networks = self.load_network()
        self.evaluator = Evaluator(self.networks, self.txt_emb)
        
        # visualizer to visdom server
        self.vis = Visualizer('http://bvisionserver9.cs.unc.edu', 8088, output_dir)
        self.vis.make_img_window("real_im")
        self.vis.make_img_window("fake_im")
        self.vis.make_txt_window("real_captions")
        self.vis.make_txt_window("genr_captions")        
        self.vis.make_plot_window("G_loss", num=7, 
                                  legend=["errG", "uncond", "cond", "latent", "cycltxt", "autoimg", "autotxt"])
        self.vis.make_plot_window("D_loss", num=4, 
                                  legend=["errD", "uncond", "cond", "latent"])
        self.vis.make_plot_window("KL_loss", num=4, 
                                  legend=["kl", "img", "txt", "fakeimg"])
        
        self.vis.make_plot_window("inception_score", num=2,
                                 legend=["real", "fake"])
        self.vis.make_plot_window("r_precision", num=1)
              
    #
    # convert a text sentence into indices
    #
    def ind_from_sent(self, caption):
        s = [self.txt_dico.SOS_TOKEN]
        s += [self.txt_dico.word2id[word] \
            if word in self.txt_dico.word2id \
            else self.txt_dico.UNK_TOKEN for word in caption.split(" ")]
        s += [self.txt_dico.EOS_TOKEN]
        return s
    
    #
    # pad a sequence with PAD TOKENs
    #
    def pad_seq(self, seq, max_length):
        seq += [self.txt_dico.PAD_TOKEN for i in range(max_length - len(seq))]
        return seq 
    
    #
    # convert a list of sentences into a padded tensor w lengths
    #
    def process_captions(self, captions):
        seqs = []
        for i in range(len(captions)):
            seqs.append(self.ind_from_sent(captions[i]))
                        
        input_lengths = [len(s) for s in seqs]
        padded = [self.pad_seq(s, max(input_lengths)) for s in seqs]
                        
        input_var = Variable(torch.LongTensor(padded)).transpose(0, 1)
        lengths = torch.LongTensor(input_lengths)
        if cfg.CUDA:
            input_var = input_var.cuda()
            lengths = lengths.cuda()
        return input_var, lengths

    #
    # load model components
    #
    def load_network(self):
        
        image_generator = ImageGenerator()
        image_generator.apply(weights_init)
        
        disc_image = DiscriminatorImage()
        disc_image.apply(weights_init)
        
        emb_dim = 300
        text_encoder = TextEncoder(emb_dim, self.txt_emb,
                         1, dropout=0.0)
        
        attn_model = 'general'
        text_generator = TextGenerator(attn_model, emb_dim, len(self.txt_dico.id2word), 
                                      self.txt_emb,
                                      n_layers=1, dropout=0.0)    
        
        image_encoder = ImageEncoder()
        image_encoder.apply(weights_init)
        
        disc_latent = DiscriminatorLatent(emb_dim)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
            
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
            
        if cfg.ENCODER != '':
            state_dict = \
                torch.load(cfg.ENCODER,
                           map_location=lambda storage, loc: storage)
            encoder.load_state_dict(state_dict)
            print('Load from: ', cfg.ENCODER)
            
        if cfg.DECODER != '':
            state_dict = \
                torch.load(cfg.DECODER,
                           map_location=lambda storage, loc: storage)
            decoder.load_state_dict(state_dict)
            print('Load from: ', cfg.DECODER)
            
        if cfg.IMAGE_ENCODER != '':
            state_dict = \
                torch.load(cfg.IMAGE_ENCODER,
                           map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load from: ', cfg.IMAGE_ENCODER)         
            
        if cfg.CUDA:
            image_encoder.cuda()
            image_generator.cuda()
            text_encoder.cuda()
            text_generator.cuda()
            disc_image.cuda()
            disc_latent.cuda()
            
        return image_encoder, image_generator, text_encoder, text_generator, disc_image, disc_latent
    
    def define_optimizers(self, 
                          image_encoder, image_generator, 
                          text_encoder, text_generator, 
                          disc_image, disc_latent):

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        
        optim_disc_img = \
            optim.Adam(disc_image.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
            
        img_gen_params = filter(lambda p: p.requires_grad, image_generator.parameters())
        optim_img_gen = optim.Adam(img_gen_params,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        
        optim_fn, optim_params = get_optimizer("adam,lr=0.001")
        enc_params = filter(lambda p: p.requires_grad, text_encoder.parameters())
        optim_txt_enc = optim_fn(enc_params, **optim_params)
        
        optim_fn, optim_params = get_optimizer("adam,lr=0.001")
        dec_params = filter(lambda p: p.requires_grad, text_generator.parameters())
        optim_txt_gen = optim_fn(dec_params, **optim_params)
        
        
        optim_img_enc = \
            optim.SGD(image_encoder.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR)
            
        optim_disc_latent = \
            optim.Adam(disc_latent.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
            
        return optim_img_enc, optim_img_gen, \
                optim_txt_enc, optim_txt_gen, \
                optim_disc_img, optim_disc_latent
                        
    #
    # train with both autoencoding and cross-domain losses
    #
    def train(self, data_loader, dataset, stage=1):
        
        image_encoder, image_generator, text_encoder, text_generator, disc_image, disc_latent = self.networks
                           
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
                     volatile=True)
            
        #
        # make labels for real/fake
        #
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # try discriminator smoothing
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        
        txt_enc_labels = Variable(torch.FloatTensor(batch_size).fill_(0)) 
        img_enc_labels = Variable(torch.FloatTensor(batch_size).fill_(1)) 
        
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
            txt_enc_labels = txt_enc_labels.cuda()
            img_enc_labels = img_enc_labels.cuda()                

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        
        optims = self.define_optimizers(image_encoder, image_generator, 
                                   text_encoder, text_generator, 
                                   disc_image, disc_latent)
        optim_img_enc, optim_img_gen, optim_txt_enc, optim_txt_gen, optim_disc_img, optim_disc_latent = optims
        
        count = 0
                
        for epoch in range(self.max_epoch):
            
            start_t = time.time()
            
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.75
                for param_group in optim_img_gen.param_groups:
                    param_group['lr'] = generator_lr
                    
                discriminator_lr *= 0.75
                for param_group in optim_disc_img.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                _, real_img_cpu, _, captions, pred_cap = data

                raw_inds, raw_lengths = self.process_captions(captions)
                
                inds, lengths = raw_inds.data, raw_lengths
                
                inds = Variable(inds)
                lens_sort, sort_idx = lengths.sort(0, descending=True)
                                
                # need to dataparallel the encoders?
                txt_encoder_output = text_encoder(inds[:, sort_idx], lens_sort.cpu().numpy(), None)
                encoder_out, encoder_hidden, real_txt_code, real_txt_mu, real_txt_logvar = txt_encoder_output
                
                real_imgs = Variable(real_img_cpu)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()

                #######################################################
                # (2) Generate fake images and their latent codes
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (real_txt_code, noise)
                fake_imgs = \
                    nn.parallel.data_parallel(image_generator, inputs, self.gpus)
                                        
                fake_img_out = nn.parallel.data_parallel(
                    image_encoder, (fake_imgs), self.gpus
                )
            
                fake_img_feats, fake_img_emb, fake_img_code, fake_img_mu, fake_img_logvar = fake_img_out
                fake_img_feats = fake_img_feats.transpose(0,1)                    
                    
                #######################################################
                # (2b) Calculate auto encoding loss for text
                ######################################################           
                loss_auto_txt, _ = compute_text_gen_loss(text_generator, 
                                                      inds[:,sort_idx],
                                                      real_txt_code.unsqueeze(0), 
                                                      encoder_out, 
                                                      self.txt_dico)
                loss_auto_txt = loss_auto_txt / lengths.float().sum() 

                #######################################################
                # (2c) Decode z from real imgs and calc auto-encoding loss
                ######################################################                    
                
                real_img_out = nn.parallel.data_parallel(
                    image_encoder, (real_imgs[sort_idx]), self.gpus
                )
                
                real_img_feats, real_img_emb, real_img_code, real_img_mu, real_img_logvar = real_img_out

                noise.data.normal_(0, 1)
                loss_auto_img, _ = compute_image_gen_loss(image_generator, 
                                                       real_imgs[sort_idx],
                                                       real_img_code,
                                                       noise,
                                                       self.gpus)
                
                #######################################################
                # (2c) Decode z from fake imgs and calc cycle loss
                ######################################################                    
                
                loss_cycle_text, gen_captions = compute_text_gen_loss(text_generator, 
                                                        inds[:,sort_idx], 
                                                        fake_img_code.unsqueeze(0), 
                                                        fake_img_feats, 
                                                        self.txt_dico)

                loss_cycle_text = loss_cycle_text / lengths.float().sum()
                
                ###############################################################
                # (2d) Generate image from predicted cap, calc img cycle loss
                ###############################################################
                
                loss_cycle_img = 0
#                 if (len(pred_cap)):
#                     pred_inds, pred_lens = pred_cap
#                     pred_inds = Variable(pred_inds.transpose(0,1))
#                     pred_inds = pred_inds.cuda() if cfg.CUDA else pred_inds

#                     pred_output = encoder(pred_inds[:, sort_idx], pred_lens.cpu().numpy(), None)
#                     pred_txt_out, pred_txt_hidden, pred_txt_code, pred_txt_mu, pred_txt_logvar = pred_output
                    
#                     noise.data.normal_(0, 1)
#                     inputs = (pred_txt_code, noise)
#                     _, fake_from_fake_img, mu, logvar = \
#                         nn.parallel.data_parallel(netG, inputs, self.gpus)
                    
#                     pred_img_out = nn.parallel.data_parallel(
#                         image_encoder, (fake_from_fake_img), self.gpus
#                     )                    
                    
#                     pred_img_feats, pred_img_emb, pred_img_code, pred_img_mu, pred_img_logvar = pred_img_out
                    
#                     semantic_target = Variable(torch.ones(batch_size))
#                     if cfg.CUDA:
#                         semantic_target = semantic_target.cuda()
                                                
#                     loss_cycle_img = cosine_emb_loss(
#                         pred_img_feats.contiguous().view(batch_size, -1), real_img_feats.contiguous().view(batch_size, -1), semantic_target
#                     )
                
                ###########################
                # (3) Update D network
                ###########################
                optim_disc_img.zero_grad()
                optim_disc_latent.zero_grad()
                
                errD = 0
                
                errD_fake_imgs = compute_cond_discriminator_loss(disc_image, fake_imgs, 
                                                   fake_labels, encoder_hidden[0], self.gpus)               
                
                errD_im, errD_real, errD_fake = \
                    compute_uncond_discriminator_loss(disc_image, real_imgs, fake_imgs,
                                                      real_labels, fake_labels,
                                                      self.gpus)
                    
                err_latent_disc = compute_latent_discriminator_loss(disc_latent, 
                                                                    real_img_emb, encoder_hidden[0],
                                                                    img_enc_labels, txt_enc_labels,
                                                                    self.gpus)
                
                # if (len(pred_cap)):
                #     errD_fake_from_fake_imgs = compute_cond_disc(netD, fake_from_fake_img, 
                #                                                  fake_labels, pred_txt_hidden[0], self.gpus)
                #     errD += errD_fake_from_fake_imgs                 
                
                errD = errD + errD_im + errD_fake_imgs + err_latent_disc
                
                # check NaN
                if (errD != errD).data.any():
                    print("NaN detected (discriminator)")
                    pdb.set_trace()
                    exit()
                    
                errD.backward()
                #temp_errD = errD_fake_imgs + errD_im
                #temp_errD.backward()
                                
                optim_disc_img.step()
                optim_disc_latent.step()
                
                # for n,p in disc_image.named_parameters():
                #     if "encode_img" in n:
                #         print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))                
                
                ############################
                # (2) Update G network
                ###########################
                optim_img_enc.zero_grad()
                optim_img_gen.zero_grad()
                optim_txt_enc.zero_grad()
                optim_txt_gen.zero_grad()
                
                errG_total = 0
                
                err_g_uncond_loss = compute_uncond_generator_loss(disc_image, fake_imgs,
                                              real_labels, self.gpus)
                
                err_g_cond_disc_loss = compute_cond_generator_loss(disc_image, fake_imgs, 
                                                                   real_labels, encoder_hidden[0], self.gpus)
                                    
                err_latent_gen = compute_latent_generator_loss(disc_latent, 
                                                               real_img_emb, encoder_hidden[0],
                                                               img_enc_labels, txt_enc_labels,
                                                               self.gpus)
                
                errG = err_g_uncond_loss + err_g_cond_disc_loss + err_latent_gen + \
                        loss_cycle_text + \
                        loss_auto_img + \
                        loss_auto_txt
                
                # if (len(pred_cap)):
                #     errG_fake_from_fake_imgs = compute_cond_disc(netD, fake_from_fake_img, 
                #                                                  real_labels, pred_txt_hidden[0], self.gpus)
                #     errG += errG_fake_from_fake_imgs                
                
                img_kl_loss = KL_loss(real_img_mu, real_img_logvar)
                txt_kl_loss = KL_loss(real_txt_mu, real_txt_logvar)
                f_img_kl_loss = KL_loss(fake_img_mu, fake_img_logvar)

                kl_loss = img_kl_loss + txt_kl_loss + f_img_kl_loss
                           
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                
                # check NaN
                if (errG_total != errG_total).data.any():
                    print("NaN detected (generator)")
                    pdb.set_trace()
                    exit()
                
                errG_total.backward()
                #temp_errG = err_g_uncond_loss + err_g_cond_disc_loss + img_kl_loss
                #temp_errG.backward()
                
                optim_img_enc.step()
                optim_img_gen.step()
                optim_txt_enc.step()
                optim_txt_gen.step()               
                
                count = count + 1
                if i % 100 == 0:
                    self.vis.add_to_plot("D_loss", np.asarray([[
                                                    errD.data[0],
                                                    errD_im.data[0],
                                                    errD_fake_imgs.data[0],
                                                    err_latent_disc.data[0]
                                                    ]]), 
                                                    np.asarray([[count] * 4]))
                    self.vis.add_to_plot("G_loss", np.asarray([[
                                                    errG.data[0], 
                                                    err_g_uncond_loss.data[0],
                                                    err_g_cond_disc_loss.data[0],
                                                    err_latent_gen.data[0],
                                                    loss_cycle_text.data[0],
                                                    loss_auto_img.data[0],
                                                    loss_auto_txt.data[0]
                                                    ]]),
                                                    np.asarray([[count] * 7]))
                    self.vis.add_to_plot("KL_loss", np.asarray([[
                                                    kl_loss.data[0],
                                                    img_kl_loss.data[0],
                                                    txt_kl_loss.data[0],
                                                    f_img_kl_loss.data[0]
                                                    ]]), 
                                         np.asarray([[count] * 4]))
                
                    self.vis.show_images("real_im", real_imgs[sort_idx].data.cpu().numpy())
                    self.vis.show_images("fake_im", fake_imgs.data.cpu().numpy())
                    
                    sorted_captions = [captions[i] for i in sort_idx.cpu().tolist()]
                    gen_cap_text = []
                    for d_i, d in enumerate(gen_captions):
                        s = u""
                        for i in d:
                            if i == self.txt_dico.EOS_TOKEN:
                                break
                            if i != self.txt_dico.SOS_TOKEN:
                                s += self.txt_dico.id2word[i] + u" "
                        gen_cap_text.append(s)
                        
                    self.vis.show_text("real_captions", sorted_captions)
                    self.vis.show_text("genr_captions", gen_cap_text)
                    
                    r_precision = self.evaluator.r_precision_score(fake_img_code, real_txt_code)
                    self.vis.add_to_plot("r_precision", np.asarray([r_precision.data[0]]), np.asarray([count]))
                                                        
                        
#             # save pred caps for next iteration
#             for i, data in enumerate(data_loader, 0):
#                 keys, real_img_cpu, _, _, _ = data
#                 real_imgs = Variable(real_img_cpu)
#                 if cfg.CUDA:
#                     real_imgs = real_imgs.cuda()                
                
#                 cap_img_out = nn.parallel.data_parallel(
#                     image_encoder, (real_imgs[sort_idx]), self.gpus
#                 )
                
#                 cap_img_feats, cap_img_emb, cap_img_code, cap_img_mu, cap_img_logvar = cap_img_out
#                 cap_img_feats = cap_img_feats.transpose(0,1)
                                                
#                 cap_features = cap_img_code.unsqueeze(0)
                
#                 cap_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
#                 cap_dec_inp = cap_dec_inp.cuda() if cfg.CUDA else cap_dec_inp

#                 cap_dec_hidden = cap_features.detach()

#                 seq = torch.LongTensor([])
#                 seq = seq.cuda() if cfg.CUDA else seq

#                 max_target_length = 20
                
#                 lengths = torch.LongTensor(batch_size).fill_(20)

#                 for t in range(max_target_length):

#                     cap_dec_out, cap_dec_hidden, cap_dec_attn = decoder(
#                         cap_dec_inp, cap_dec_hidden, cap_img_feats
#                     )

#                     topv, topi = cap_dec_out.topk(1, dim=1)

#                     cap_dec_inp = topi #.squeeze(dim=2)
#                     cap_dec_inp = cap_dec_inp.cuda() if cfg.CUDA else cap_dec_inp

#                     seq = torch.cat((seq, cap_dec_inp.data), dim=1)

#                 dataset.save_captions(keys, seq.cpu(), lengths.cpu())

            #self.evaluator.run_all_eval()
            
            iscore_mu_real, _ = self.evaluator.inception_score(real_imgs[sort_idx])
            iscore_mu_fake, _ = self.evaluator.inception_score(fake_imgs)
            self.vis.add_to_plot("inception_score", np.asarray([[
                        iscore_mu_real,
                        iscore_mu_fake
                    ]]),
                    np.asarray([[epoch] * 2]))    
            
            end_t = time.time()
            
            prefix = "Epoch %d; %s, %.1f sec" % (epoch, time.strftime('D%d %X'), (end_t-start_t))
            gen_str = "G_total: %.3f Gen loss: %.3f KL loss %.3f" % (
                                                                         errG_total.data[0],
                                                                         errG.data[0],
                                                                         kl_loss.data[0]
                                                                        )
            
            dis_str = "Img Disc: %.3f Latent Disc: %.3f" % (
                errD.data[0], 
                err_latent_disc.data[0]
            )
            
            eval_str = "Incep real: %.3f Incep fake: %.3f R prec %.3f" % (
                iscore_mu_real, 
                iscore_mu_fake,
                r_precision
            )
                
            print("%s %s, %s; %s" % (prefix, gen_str, dis_str, eval_str))
            
            if epoch % self.snapshot_interval == 0:
                save_model(image_encoder, image_generator, 
                           text_encoder, text_generator, 
                           disc_image, disc_latent,
                           epoch, self.model_dir)

        save_model(image_encoder, image_generator, 
                   text_encoder, text_generator, 
                   disc_image, disc_latent, 
                   epoch, self.model_dir)
        
        self.summary_writer.close()
    
    def sample(self, data_loader, stage=1):
        print("todo")