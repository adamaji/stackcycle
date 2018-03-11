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

import pickle

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from miscc.utils import compute_discriminator_loss_cycle

from miscc.utils import load_external_embeddings, get_optimizer

from tensorboardX import summary
from tensorboardX import FileWriter

from nltk import word_tokenize

class GANTrainer(object):
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
        
        path = "../data/birds/birds.en.vec"
        txt_dico, _txt_emb = load_external_embeddings(path)
        #params.src_dico = src_dico
        txt_emb = nn.Embedding(len(txt_dico), 300, sparse=False)
        txt_emb.weight.data.copy_(_txt_emb)
        txt_emb.weight.requires_grad = False
        self.txt_dico = txt_dico
        self.txt_emb = txt_emb
        
    def ind_from_sent(self, caption):
        return [self.txt_dico.SOS_TOKEN] + [self.txt_dico.word2id[word] if word in self.txt_dico.word2id else self.txt_dico.UNK_TOKEN for word in caption.split(" ")] + [self.txt_dico.EOS_TOKEN]
    
    def pad_seq(self, seq, max_length):
        seq += [self.txt_dico.PAD_TOKEN for i in range(max_length - len(seq))]
        return seq 
    
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
    
    def add_noise(self, inds, lens):

        pwd = 0.1
        k = 3
        
        inds = inds.transpose(0,1)
        
        mask = torch.rand(inds.size()) > pwd
        if cfg.CUDA: mask = mask.cuda()
        mask = mask
    
        max_len = inds.size(1)
        masked = torch.masked_select(inds, mask)
        chopped_lens = torch.sum(mask, dim=1)
        i = 0
        seq = []
        for cl in chopped_lens:
            zeros = torch.zeros(max_len - cl).long() # should this be padding?
            zeros = zeros.cuda() if cfg.CUDA else zeros
            seq.append(torch.cat((masked[i:i+cl],zeros)))
            i += cl
        seq = torch.stack(seq)
                    
        # get sequence lengths
        EOS = self.txt_dico.EOS_TOKEN
            
        seq_lens = []
        eos_inds = torch.nonzero(seq == EOS)

        # in case there are no predicted EOS
        for b_idx in range(lens.size(0)):
            if eos_inds.size() == () or b_idx not in eos_inds[:,0]:
                app = torch.cuda.LongTensor([b_idx, 20]) if cfg.CUDA else torch.LongTensor([b_idx, 20])
                eos_inds = torch.cat((eos_inds, app.unsqueeze(0)), dim=0)  
                
        ind = -1
        for s in eos_inds:
            if s[0] != ind:
                if s[1] == 0: # HACK TO MAKE SENTS WITH EOS FIRST IND HAVE NONZERO LEN
                    seq_lens.append(s[1] + 1)
                else:
                    seq_lens.append(s[1])
                ind = s[0]
            
        # permute words in window of k
        for b_idx, s in enumerate(seq):
            l = seq_lens[b_idx] - 1 # 1 for EOS
            for i in range(1,l,k): # skip SOS
                ki = k if i+k < l else l-i
                p = torch.randperm(ki) + i
                p = p.cuda() if cfg.CUDA else p
                seq[b_idx,i:ki+i] = s[p]
                
        seq_lens = torch.cuda.LongTensor(seq_lens) if cfg.CUDA else torch.LongTensor(seq_lens)
        return seq.transpose(0,1), seq_lens    

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D, EncoderRNN, LuongAttnDecoderRNN, STAGE1_ImageEncoder
        netG = STAGE1_G()
        netG.apply(weights_init)
        #print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        #print(netD)
        
        emb_dim = 300
        encoder = EncoderRNN(emb_dim, self.txt_emb,
                         1, dropout=0.0)
        
        attn_model = 'general'
        decoder = LuongAttnDecoderRNN(attn_model, emb_dim, len(self.txt_dico.id2word), 
                                      self.txt_emb,
                                      2, dropout=0.0)    
        
        image_encoder = STAGE1_ImageEncoder()
        image_encoder.apply(weights_init)

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
            netG.cuda()
            netD.cuda()
            encoder.cuda()
            decoder.cuda()
            image_encoder.cuda()
            
        return netG, netD, encoder, decoder, image_encoder

    # ############# For training stageII GAN  #############
#     def load_network_stageII(self):
#         from model import STAGE1_G, STAGE2_G, STAGE2_D

#         Stage1_G = STAGE1_G()
#         netG = STAGE2_G(Stage1_G)
#         netG.apply(weights_init)
#         print(netG)
#         if cfg.NET_G != '':
#             state_dict = \
#                 torch.load(cfg.NET_G,
#                            map_location=lambda storage, loc: storage)
#             netG.load_state_dict(state_dict)
#             print('Load from: ', cfg.NET_G)
#         elif cfg.STAGE1_G != '':
#             state_dict = \
#                 torch.load(cfg.STAGE1_G,
#                            map_location=lambda storage, loc: storage)
#             netG.STAGE1_G.load_state_dict(state_dict)
#             print('Load from: ', cfg.STAGE1_G)
#         else:
#             print("Please give the Stage1_G path")
#             return

#         netD = STAGE2_D()
#         netD.apply(weights_init)
#         if cfg.NET_D != '':
#             state_dict = \
#                 torch.load(cfg.NET_D,
#                            map_location=lambda storage, loc: storage)
#             netD.load_state_dict(state_dict)
#             print('Load from: ', cfg.NET_D)
#         print(netD)

#         if cfg.CUDA:
#             netG.cuda()
#             netD.cuda()
#         return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD, encoder, decoder, image_encoder = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
                     volatile=True)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # try discriminator smoothing
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        
        optim_fn, optim_params = get_optimizer("adam,lr=0.001")
        enc_params = filter(lambda p: p.requires_grad, encoder.parameters())
        enc_optimizer = optim_fn(enc_params, **optim_params)
        optim_fn, optim_params = get_optimizer("adam,lr=0.001")
        dec_params = filter(lambda p: p.requires_grad, decoder.parameters())
        dec_optimizer = optim_fn(dec_params, **optim_params)        
        
        image_enc_optimizer = \
            optim.Adam(image_encoder.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))        
        
        count = 0
        
        criterionCycle = torch.nn.L1Loss()
        
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_img_cpu, txt_embedding, captions = data
                raw_inds, raw_lengths = self.process_captions(captions)
                inds, lengths = self.add_noise(raw_inds.data, raw_lengths)
                inds = Variable(inds)
                lens_sort, sort_idx = lengths.sort(0, descending=True)
                encoder_out, encoder_hidden = encoder(inds[:, sort_idx], lens_sort.cpu().numpy(), None)
                
                real_imgs = Variable(real_img_cpu)
                #txt_embedding = Variable(txt_embedding)
                
                txt_embedding = encoder_hidden[0]
                
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)
                    
                #######################################################
                # (2b) Decode z from txt and calc auto-encoding loss
                ######################################################                    
                loss_auto = 0
                auto_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
                auto_dec_inp = auto_dec_inp.cuda() if cfg.CUDA else auto_dec_inp
                auto_dec_hidden = encoder_hidden[:decoder.n_layers]

                max_target_length = 20

                for t in range(max_target_length):

                    auto_dec_out, auto_dec_hidden, auto_dec_attn = decoder(
                        auto_dec_inp, auto_dec_hidden, encoder_out
                    )

                    loss_auto = loss_auto + F.cross_entropy(auto_dec_out, 
                                                            inds[:,sort_idx][t], ignore_index=self.txt_dico.PAD_TOKEN)
                    auto_dec_inp = inds[:,sort_idx][t]

                loss_auto = loss_auto / lengths.float().sum()  

                #######################################################
                # (2c) Decode z from real imgs and calc auto-encoding loss
                ######################################################                    
                
                real_features = nn.parallel.data_parallel(image_encoder, (real_imgs), self.gpus)
                             
                noise.data.normal_(0, 1)
                inputs = (real_features[0], noise)
                _, fake_from_real_img, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)            
                
                loss_img = criterionCycle(fake_from_real_img, real_imgs)
                
                #######################################################
                # (2c) Decode z from fake imgs and calc cycle loss
                ######################################################                    
                
                fake_features = nn.parallel.data_parallel(image_encoder, (fake_imgs), self.gpus)
             
                loss_cd = 0
                cd_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
                cd_dec_inp = cd_dec_inp.cuda() if cfg.CUDA else cd_dec_inp
                
                cd_dec_hidden = fake_features # check feat length

                max_target_length = 20

                for t in range(max_target_length):

                    cd_dec_out, cd_dec_hidden, cd_dec_attn = decoder(
                        cd_dec_inp, cd_dec_hidden, encoder_out
                    )

                    loss_cd = loss_cd + F.cross_entropy(cd_dec_out, inds[:,sort_idx][t], ignore_index=self.txt_dico.PAD_TOKEN)
                    cd_dec_inp = inds[:,sort_idx][t]

                loss_cd = loss_cd / lengths.float().sum()                  
                
                #######################################################
                # (2d) Update encoder/decoder
                ######################################################                  
                encoder.zero_grad()
                decoder.zero_grad()
                
                loss_auto.backward(retain_graph=True)
                
                enc_optimizer.step()
                dec_optimizer.step()
                
                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                # errD, errD_real, errD_wrong, errD_fake = \
                #     compute_discriminator_loss(netD, real_imgs, fake_imgs,
                #                                real_labels, fake_labels,
                #                                mu, self.gpus)
                errD, errD_real, errD_fake = \
                    compute_discriminator_loss_cycle(netD, real_imgs, fake_imgs,
                                                     real_labels, fake_labels,
                                                     mu, self.gpus)
                errD.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                image_encoder.zero_grad()
                
                errG = compute_generator_loss(netD, fake_imgs,
                                              real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL + loss_cd + loss_img
                errG_total.backward()
                
                optimizerG.step()
                image_enc_optimizer.step()

                count = count + 1
                if i % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD.data[0])
                    summary_D_r = summary.scalar('D_loss_real', errD_real)
                    #summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    summary_G = summary.scalar('G_loss', errG.data[0])
                    summary_KL = summary.scalar('KL_loss', kl_loss.data[0])

                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_D_r, count)
                    #self.summary_writer.add_summary(summary_D_w, count)
                    self.summary_writer.add_summary(summary_D_f, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                    # save the image result for each epoch
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)
                    #save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    #if lr_fake is not None:
                    #    save_img_results(None, lr_fake, epoch, self.image_dir)
                        
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f Loss_cd: %.4f
                     Loss_real: %.4f Loss_fake %.4f Loss_auto_enc: %.4f Loss_img: %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.data[0], errG.data[0], kl_loss.data[0], loss_cd.data[0],
                     errD_real, errD_fake, loss_auto.data[0], loss_img.data[0], (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, encoder, decoder, image_encoder, epoch, self.model_dir)
        #
        save_model(netG, netD, encoder, decoder, image_encoder, self.max_epoch, self.model_dir)
        #
        self.summary_writer.close()

    def sample(self, data_loader, stage=1):
        if stage == 1:
            netG, _, encoder, decoder, image_encoder = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # for i, data in enumerate(data_loader, 0):
        #     real_img_cpu, txt_embedding, captions = data
        
        with open("test128.pkl", "rb") as f:
            data = pickle.load(f)
            #pickle.dump(data, f)

        real_img_cpu, txt_embedding, captions = data
        real_imgs = Variable(real_img_cpu)
        #txt_embedding = Variable(txt_embedding)

        inds, lengths = self.process_captions(captions)
        lens_sort, sort_idx = lengths.sort(0, descending=True)
        encoder_out, encoder_hidden = encoder(inds[:, sort_idx], lens_sort.cpu().numpy(), None)
        txt_embedding = encoder_hidden[0]
        
        sorted_captions = [captions[i] for i in sort_idx.cpu().tolist()]
        
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()
            txt_embedding = txt_embedding.cuda()            
            # if i>0:
            #     break
                                
        # Load text embeddings generated from the encoder
        #t_file = torchfile.load(datapath)
        #captions_list = t_file.raw_txt
        #embeddings = np.concatenate(t_file.fea_txt, axis=0)
        #num_embeddings = len(captions_list)
        #print('Successfully load sentences from: ', datapath)
        #print('Total number of sentences:', num_embeddings)
        #print('num_embeddings:', num_embeddings, embeddings.shape)
        
        # path to save generated samples
        #save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        save_dir = "/playpen1/aji/stackcycle_results/%s/" % (cfg.DATASET_NAME)
        mkdir_p(save_dir)

        #batch_size = np.minimum(num_embeddings, self.batch_size)
        batch_size = self.batch_size
        num_embeddings = batch_size
        
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        # count = 0
        # while count < num_embeddings:
        #     if count > 3000:
        #         break
        #     iend = count + batch_size
        #     if iend > num_embeddings:
        #         iend = num_embeddings
        #         count = num_embeddings - batch_size
        #     embeddings_batch = embeddings[count:iend]
        #     # captions_batch = captions_list[count:iend]
        #     txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
        #     if cfg.CUDA:
        #         txt_embedding = txt_embedding.cuda()
        
        #######################################################
        # auto encoding
        #######################################################
        
        auto_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
        auto_dec_inp = auto_dec_inp.cuda() if cfg.CUDA else auto_dec_inp
        auto_dec_hidden = encoder_hidden[:decoder.n_layers]
        
        seq = torch.LongTensor([])
        seq = seq.cuda() if cfg.CUDA else seq          

        max_target_length = 20

        for t in range(max_target_length):

            auto_dec_out, auto_dec_hidden, auto_dec_attn = decoder(
                auto_dec_inp, auto_dec_hidden, encoder_out
            )
            
            topv, topi = auto_dec_out.topk(1, dim=1)
                        
            auto_dec_inp = topi #.squeeze(dim=2)
            auto_dec_inp = auto_dec_inp.cuda() if cfg.CUDA else auto_dec_inp

            seq = torch.cat((seq, auto_dec_inp.data), dim=1)               

            #auto_dec_inp = inds[:,sort_idx][t]
            
        auto_captions = []
        for d_i, d in enumerate(seq):
            s = u""
            for i in d:
                if i == self.txt_dico.EOS_TOKEN:
                    break
                if i != self.txt_dico.SOS_TOKEN:
                    s += unicode(self.txt_dico.id2word[i].decode("utf-8")) + u" "
            auto_captions.append(s)
            
        save_name = "%s/auto_encoded.txt" % (save_dir)
        with open(save_name, "w") as f:
            for i in range(batch_size):
                f.write("ORIG %d\t%s\nAUTO %d\t%s\n\n" % (i, sorted_captions[i], i, auto_captions[i]))
            

        #######################################################
        # (2) Generate fake images
        ######################################################
        noise.data.normal_(0, 1)
        inputs = (txt_embedding, noise)
        _, fake_imgs, mu, logvar = \
            nn.parallel.data_parallel(netG, inputs, self.gpus)
                        
        for i in range(batch_size):
            save_name = '%s/fake_%03d.png' % (save_dir, i)
            im = fake_imgs[i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            im.save(save_name)
            # count += batch_size
            
        for i in range(batch_size):
            save_name = '%s/real_%03d.png' % (save_dir, i)
            im = real_imgs[sort_idx][i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            # print('im', im.shape)
            im = np.transpose(im, (1, 2, 0))
            # print('im', im.shape)
            im = Image.fromarray(im)
            im.save(save_name)
            # count += batch_size            
            
        # write original captions
            
        save_name = "%s/captions.txt" % (save_dir)
        with open(save_name, "w") as f:
            for i in range(batch_size):
                f.write("%d\t%s\n" % (i, sorted_captions[i]))
        print("Saved to %s" % save_dir)
        
        #######################################################
        # cycle
        #######################################################
        
        fake_features = nn.parallel.data_parallel(image_encoder, (fake_imgs), self.gpus)
        
        cy_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
        cy_dec_inp = cy_dec_inp.cuda() if cfg.CUDA else cy_dec_inp
        #cy_dec_hidden = encoder_hidden[:decoder.n_layers]
        
        cy_dec_hidden = fake_features
        
        seq = torch.LongTensor([])
        seq = seq.cuda() if cfg.CUDA else seq          

        max_target_length = 20

        for t in range(max_target_length):

            cy_dec_out, cy_dec_hidden, cy_dec_attn = decoder(
                cy_dec_inp, cy_dec_hidden, encoder_out
            )
            
            topv, topi = cy_dec_out.topk(1, dim=1)
                        
            cy_dec_inp = topi #.squeeze(dim=2)
            cy_dec_inp = cy_dec_inp.cuda() if cfg.CUDA else cy_dec_inp

            seq = torch.cat((seq, cy_dec_inp.data), dim=1)               

            #auto_dec_inp = inds[:,sort_idx][t]
            
        cy_captions = []
        for d_i, d in enumerate(seq):
            s = u""
            for i in d:
                if i == self.txt_dico.EOS_TOKEN:
                    break
                if i != self.txt_dico.SOS_TOKEN:
                    s += unicode(self.txt_dico.id2word[i].decode("utf-8")) + u" "
            cy_captions.append(s)
            
        save_name = "%s/cycle_encoded.txt" % (save_dir)
        with open(save_name, "w") as f:
            for i in range(batch_size):
                f.write("ORIG %d\t%s\nCYCL %d\t%s\n\n" % (i, sorted_captions[i], i, cy_captions[i]))        

        #######################################################
        # real image captioning
        #######################################################
        
        real_features = nn.parallel.data_parallel(image_encoder, (real_imgs[sort_idx]), self.gpus)
        
        cap_dec_inp = Variable(torch.LongTensor([self.txt_dico.SOS_TOKEN] * self.batch_size))
        cap_dec_inp = cap_dec_inp.cuda() if cfg.CUDA else cap_dec_inp
        
        cap_dec_hidden = real_features
        
        seq = torch.LongTensor([])
        seq = seq.cuda() if cfg.CUDA else seq          

        max_target_length = 20

        for t in range(max_target_length):

            cap_dec_out, cap_dec_hidden, cap_dec_attn = decoder(
                cap_dec_inp, cap_dec_hidden, encoder_out
            )
            
            topv, topi = cap_dec_out.topk(1, dim=1)
                        
            cap_dec_inp = topi #.squeeze(dim=2)
            cap_dec_inp = cap_dec_inp.cuda() if cfg.CUDA else cap_dec_inp

            seq = torch.cat((seq, cap_dec_inp.data), dim=1)               

            #auto_dec_inp = inds[:,sort_idx][t]
            
        cap_captions = []
        for d_i, d in enumerate(seq):
            s = u""
            for i in d:
                if i == self.txt_dico.EOS_TOKEN:
                    break
                if i != self.txt_dico.SOS_TOKEN:
                    s += unicode(self.txt_dico.id2word[i].decode("utf-8")) + u" "
            cap_captions.append(s)
            
        save_name = "%s/realim2cap_encoded.txt" % (save_dir)
        with open(save_name, "w") as f:
            for i in range(batch_size):
                f.write("ORIG %d\t%s\nPRED %d\t%s\n\n" % (i, sorted_captions[i], i, cap_captions[i]))                 
