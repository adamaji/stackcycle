from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--name', type=str, default=None)
    
    # we can make ablations more sophisticated later...
    # need:
    #  -imgcycle
    #  -txtcycle
    #  -imgauto
    #  -txtauto
    #  -spvimgtxt
    #  -spvtxtimg
    parser.add_argument('--imgcycle', type=bool, default=False)
    parser.add_argument('--txtcycle', type=bool, default=False)
    parser.add_argument('--imgauto', type=bool, default=False)
    parser.add_argument('--txtauto', type=bool, default=False)
    parser.add_argument('--spvimgtxt', type=bool, default=False)
    parser.add_argument('--spvtxtimg', type=bool, default=False)
    parser.add_argument('--disclatent', type=bool, default=False)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
        
    # ablations
    cfg.AB.imgcycle = args.imgcycle
    cfg.AB.txtcycle = args.txtcycle
    cfg.AB.imgauto = args.imgauto
    cfg.AB.txtauto = args.txtauto
    cfg.AB.spvimgtxt = args.spvimgtxt
    cfg.AB.spvtxtimg = args.spvtxtimg
    cfg.AB.disclatent = args.disclatent
        
    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    
    name = args.name if args.name else input("Enter a name for this run: ")
    
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, name)

    num_gpu = len(cfg.GPU_ID.split(','))
                        
    image_transform = transforms.Compose([
        transforms.RandomCrop(cfg.IMSIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                        
              
    # training
    if cfg.TRAIN.FLAG:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = Trainer(output_dir)
        algo.train(dataloader, dataset, cfg.STAGE)
         
    # testing
    else:    
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))        
        
        algo = Trainer(output_dir)
        algo.sample(dataloader, cfg.STAGE)
