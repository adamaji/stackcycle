import numpy as np

from code.trainer import GANTrainer

class Evaluator(object):
    
    def __init__(self, trainer):
        self.txt_emb = trainer.txt_emb
        self.txt_encoder = trainer.txt_encoder
        self.txt_generator = trainer.txt_generator
        self.img_encoder = trainer.img_encoder
        self.img_generator = trainer.img_generator
    
    ################################
    # image generation evaluations
    ################################
    
    #
    # runs inception score on txt->img generations
    #
    def inception_score(self, log):
        print("todo")
        
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
        