## stackcycle

This project attempted (unsuccessfully) to align visual and textual spaces via cyclic reconstruction. That is, we wanted to learn a shared visual-textual embedding space by generating images from text generated from images (and vice versa). 

Motivations behind this approach include results from image captioning and image generation (from text) models, as well as results from papers like [CycleGAN](https://github.com/junyanz/CycleGAN) and FAIR's [unsupervised machine translation](https://arxiv.org/abs/1711.00043). 

This code is built off of a fork of the initial pytorch implementation of [StackGAN](https://github.com/hanzhanggit/StackGAN-Pytorch) (text to image generator). The original readme is [here](README_stackgan.md).

### Data

The main datasets I used with this were the CUB 200 and Oxford 102 datasets. (Simple text-image pairs for birds and flowers respectively.) You should be able to find them [here](https://github.com/reedscot/icml2016).

### Steps to run

The steps to run are like those from the original StackGAN code:

Training: `python main.py --cfg cfg/birds_s1.yml --gpu 0` 

Evaluating: `python main.py --cfg/birds_eval.yml --gpu 0` 
(In actuality I used a couple of jupyter notebooks to run eval so I could test some different configs more quickly. They're really messy though, so I've just popped some of those routines into the evaluator.py I meant to use for now.)

