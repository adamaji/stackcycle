import pdb
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from miscc.config import cfg

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
        
        image_encoder, image_generator, text_encoder, text_generator, disc_image, disc_latent = networks
        
        self.txt_emb = txt_emb
        self.txt_encoder = text_encoder
        self.txt_generator = text_generator
        self.img_encoder = image_encoder
        self.img_generator = image_generator
    
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
        
#         noise = Variable(torch.FloatTensor(batch_size, nz))
#         fixed_noise = \
#             Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
#                      volatile=True)

#         txt_encoder_output = self.txt_encoder(caption_inds, caption_lens.cpu().numpy(), None)
#         encoder_out, encoder_hidden, real_txt_code, real_txt_mu, real_txt_logvar = txt_encoder_output
        
#         noise.data.normal_(0, 1)
#         inputs = (real_txt_code, noise)
#         fake_imgs = \
#             nn.parallel.data_parallel(self.img_generator, inputs, gpus)
            
            
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
    #
    def r_precision_score(self, log):
        print("todo")
    
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
    def run_all_eval(self, log):
        self.bleu1(log)
        