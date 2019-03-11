import pdb
import numpy as np
import os
import operator

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms

from miscc.config import cfg
from miscc.utils import load_external_embeddings
from miscc.datasets import TextDataset

# inception score
import tensorflow as tf
from .inceptionscore.inception_score import inference, get_inception_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# normalize imgs to [0,1]
def normalize255(nparr):
    mn = nparr.min()
    if mn < 0:
        nparr = nparr + -1 * mn
    mx = nparr.max()
    return nparr / mx * 255

class Evaluator(object):
    
    def __init__(self, networks, txt_emb):
        # builds an evaluator for running tests on model
        # most of this is not yet functional -- pulled from notebooks
           
        image_encoder, image_generator, text_encoder, text_generator, disc_image, disc_latent = networks
        
        self.txt_emb = txt_emb
        self.text_encoder = text_encoder
        self.text_generator = text_generator
        self.image_encoder = image_encoder
        self.image_generator = image_generator
        
        # load fasttext embeddings (e.g., birds.en.vec)
        path = os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME + ".en.vec")
        txt_dico, _txt_emb = load_external_embeddings(path)
        txt_emb = nn.Embedding(len(txt_dico), 300, sparse=False)
        txt_emb.weight.data.copy_(_txt_emb)
        txt_emb.weight.requires_grad = False
        self.txt_dico = txt_dico
        self.txt_emb = txt_emb        
        
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        
        
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=10,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
        
        self.dataloader = dataloader
        
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus] # 0        
        
        
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
    
    ################################
    # image generation evaluations
    ################################
    
    #
    # runs inception score on txt->img generations
    # assume pre-sorted
    # args: list of images
    #
    def inception_score(self, images):                
        nz = cfg.Z_DIM
        batch_size = cfg.TRAIN.BATCH_SIZE
        
        s_gpus = cfg.GPU_ID.split(',')
        gpus = [int(ix) for ix in s_gpus] # 0
        
            
        # modified from main func in inception_score.py
        
        checkpoint_dir = cfg.EVAL.INCEPTION_CKPT
        num_classes = 50 # 20 for flowers
        gpu = gpus[0] # 0
        splits = 10
        batch_size = cfg.TRAIN.BATCH_SIZE
        
        MOVING_AVERAGE_DECAY = 0.9999

        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                with tf.device("/gpu:%d" % gpu):
                    # Number of classes in the Dataset label set plus 1.
                    # Label 0 is reserved for an (unused) background class.
                    num_classes = num_classes + 1

                    # Build a Graph that computes the logits predictions from the
                    # inference model.
                    inputs = tf.placeholder(
                        tf.float32, [batch_size, 299, 299, 3],
                        name='inputs')
                    # print(inputs)

                    logits, _ = inference(inputs, num_classes)
                    # calculate softmax after remove 0 which reserve for BG
                    known_logits = \
                        tf.slice(logits, [0, 1],
                                 [batch_size, num_classes - 1])
                    pred_op = tf.nn.softmax(known_logits)

                    # Restore the moving average version of the
                    # learned variables for eval.
                    variable_averages = \
                        tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                    variables_to_restore = variable_averages.variables_to_restore()
                    saver = tf.train.Saver(variables_to_restore)
                    saver.restore(sess, checkpoint_dir)
                    #print('Restore the model from %s).' % checkpoint_dir)

                    images = normalize255(images.data.cpu().numpy())
                    mu, std = get_inception_score(sess, images, pred_op, splits, batch_size)

        return mu, std
        
    #
    # R-precision
    # args: fake images, captions used to generate them
    #
    def r_precision_score(self, images_code, captions_code, R=1, etc=99):
        
        r_precision = 0
        for im_idx, im_c in enumerate(images_code):         
            
            candidates = captions_code
            cos = F.cosine_similarity(im_c.unsqueeze(0).repeat(64,1), candidates)
            
            _, sort_idx = cos.sort(0, descending=True)
                        
            # check top-1
            r_precision += (sort_idx == im_idx)[0]
                
        r_precision = r_precision.float()
        r_precision /= images_code.size(0)
        
        return r_precision
                
            
    #
    # top-k retrieval
    # args: topk
    # TODO: pulled from notebook, need to refactor and make efficient..
    #
    def top_k_retrieval(self, topk=10):

        # count class instances to average text embeddings
        class_count = {}
        for i, data in enumerate(self.dataloader, 0):
            (key, cls_id), real_img_cpu, _, captions, pred_cap = data
            for k in key:
                idx = k.split("/")[0]
                if idx not in class_count:
                    class_count[idx] = 0
                class_count[idx] += 1

        # extract embeddings for each sentence,
        embs = {}
        for i, data in enumerate(self.dataloader, 0):
            (key, cls_id), real_img_cpu, _, captions, pred_cap = data    
            idxs = key
            
            inds, lengths = process_captions(captions)
            lens_sort, sort_idx = lengths.sort(0, descending=True)
            encoder_output = txt_encoder(inds[:, sort_idx][:,:].cuda(), lens_sort[:].cpu().numpy(), None)
            encoder_out, encoder_hidden, enc_txt_code, enc_txt_mu, enc_txt_logvar = encoder_output
            
            for b in range(len(enc_txt_code)):
                k = idxs[sort_idx[b]]
                txt_vec = encoder_out[0,b]
                if k not in embs:
                    embs[k] = txt_vec
                else:
                    embs[k] = embs[k] + txt_vec
                    pass

        # and average per class
        avg_embs = {}
        for key in embs:
            avg_embs[key] = embs[key] / class_count[key]
    

        # extract image embeddings and compare to each class embedding
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        pairs = {}
        for i, data in enumerate(self.dataloader, 0):
            (key, cls_id), real_img_cpu, _, captions, pred_cap = data
            idxs = key
            
            img_out = img_encoder(Variable(real_img_cpu[sort_idx].cuda()))
            real_img_feats, real_img_emb, real_img_code, real_img_mu, real_img_logvar = img_out   
                
            for b in range(len(real_img_code)):
                k = idxs[sort_idx[b]]
                highest = 0
                hkey = None
                
                for key in avg_embs:
                    avg = avg_embs[key]
                    
                    img_vec = real_img_feats[b,0]
                    
                    score = cos(avg, img_vec).data.cpu().numpy()[0]
                    
                    if key not in pairs: pairs[key] = []
                    pairs[key].append((k, score))

        ap_total = 0
        for k in pairs:
            pairs[k].sort(key=operator.itemgetter(1), reverse=False)
            ap = sum([p[0]==k for p in pairs[k][:int(topk)]])
            ap_total += ap / topk

        ap_k = ap_total / len(pairs)
        return ap_k

    ################################
    # image captioning evaluations
    ################################
    
    #
    # bleu@1 score from img->txt prediction vs img captions
    #
    def bleu1(self, log):
        print("todo")
        
    def bleu4(self, log):
        print("todo")
        
    def spice(self, log):
        print("todo")
    
    #
    # run all evaluations
    #
    def run_all_eval(self):
        
        # generate 30K fake images for eval
        
        all_fake_imgs = []
        all_fake_img_codes = []
        all_caption_codes = []
        
        for i, data in enumerate(self.dataloader, 0):
            
            nz = cfg.Z_DIM
            batch_size = 10
            
            if i * batch_size > 30000:
                break
            
            _, real_img_cpu, _, captions, pred_cap = data      
            
            noise = Variable(torch.FloatTensor(batch_size, nz))
            fixed_noise = \
                Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
                         volatile=True)            
            
            raw_inds, raw_lengths = self.process_captions(captions)

            inds, lengths = raw_inds.data, raw_lengths

            inds = Variable(inds)
            lens_sort, sort_idx = lengths.sort(0, descending=True)

            # need to dataparallel the encoders?
            txt_encoder_output = self.text_encoder(inds[:, sort_idx], lens_sort.cpu().numpy(), None)
            encoder_out, encoder_hidden, real_txt_code, real_txt_mu, real_txt_logvar = txt_encoder_output      
            
            real_imgs = Variable(real_img_cpu)
            if cfg.CUDA:
                real_imgs = real_imgs.cuda()   
                
            noise.data.normal_(0, 1)
            inputs = (real_txt_code, noise)
            fake_imgs = \
                nn.parallel.data_parallel(self.image_generator, inputs, self.gpus)
            fake_img_out = nn.parallel.data_parallel(
                self.image_encoder, (fake_imgs), self.gpus
            )

            fake_img_feats, fake_img_emb, fake_img_code, fake_img_mu, fake_img_logvar = fake_img_out
            fake_img_feats = fake_img_feats.transpose(0,1)     
                
            all_fake_imgs.append(fake_imgs.cpu())
            all_fake_img_codes.append(fake_img_code.cpu())
            all_caption_codes.append(real_txt_code.cpu())
        
        
