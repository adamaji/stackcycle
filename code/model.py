import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable

import pdb

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

class EncodingDiscriminator(nn.Module):

    def __init__(self, emb_dim):
        super(EncodingDiscriminator, self).__init__()

        self.emb_dim = emb_dim
        self.dis_layers = 3
        self.dis_hid_dim = 1024
        self.dis_dropout = 0.0
        self.dis_input_dropout = 0.0

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
                
        #layers.append(nn.LogSigmoid())
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #assert x.dim() == 3 and x.size(2) == self.emb_dim
        l = self.layers(x)
        #return l.view(-1)
        #return torch.exp(torch.sum(l, dim=0))
        return l

class STAGE1_ImageEncoder(nn.Module):
    def __init__(self):
        super(STAGE1_ImageEncoder, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            #nn.Linear(1024, 300)
        )
        
        # self.encode_img = nn.Sequential(
        #     nn.Linear(64 * 64 * 3, 1024),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        
        #self.linear = nn.Linear(96 * self.batch_size, 300 * 2)
        #self.linear = nn.Linear(1024, 300 * 2)
        self.l1 = nn.Linear(768 * 2 * 2, 300)
        self.l2 = nn.Linear(768 * 2 * 2, 300)

        # self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        # self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)#.view(-1, 64 * 64 * 3))
        #emb = self.linear(img_embedding.view(-1, 1024))
        mu = self.l1(img_embedding.view(-1, 768 * 2 * 2))
        logvar = self.l2(img_embedding.view(-1, 768 * 2 * 2))
        return mu, logvar

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, src_emb, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.src_emb = src_emb
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        #embedded = self.embedding(input_seqs)
        embedded = self.src_emb(input_seqs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden
    
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        # hidden -> s x b
        # encoder_outputs -> s x b
            
        energy = self.attn(encoder_outputs).transpose(0,1) # -> b x s x h
        dotted = energy.bmm(hidden.transpose(0, 1).transpose(1, 2)) # -> b x s x 1
        attn_energies = dotted.squeeze(2)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, src_emb, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        

        # Define layers
        self.src_emb = src_emb
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        #embedded = self.embedding(input_seq)
        embedded = self.src_emb(input_seq)
       
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        #self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights    

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                #nn.Sigmoid())
            )

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        #ninput = self.z_dim + self.ef_dim
        ninput = 300
        
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())
        
        self.l1 = nn.Linear(300, 1024)
        self.r1 = nn.ReLU(True)
        self.l2 = nn.Linear(1024, 64 * 64 * 3)

    def forward(self, text_embedding, noise):
        #c_code, mu, logvar = self.ca_net(text_embedding)
        #z_c_code = torch.cat((noise, c_code), 1)
        z_c_code = text_embedding
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)

        #fake_img = self.l2(self.r1(self.l1(text_embedding))).view(128, 3, 64, 64)
        
        return None, fake_img, None, None #mu, logvar


class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


# # ############# Networks for stageII GAN #############
# class STAGE2_G(nn.Module):
#     def __init__(self, STAGE1_G):
#         super(STAGE2_G, self).__init__()
#         self.gf_dim = cfg.GAN.GF_DIM
#         self.ef_dim = cfg.GAN.CONDITION_DIM
#         self.z_dim = cfg.Z_DIM
#         self.STAGE1_G = STAGE1_G
#         # fix parameters of stageI GAN
#         for param in self.STAGE1_G.parameters():
#             param.requires_grad = False
#         self.define_module()

#     def _make_layer(self, block, channel_num):
#         layers = []
#         for i in range(cfg.GAN.R_NUM):
#             layers.append(block(channel_num))
#         return nn.Sequential(*layers)

#     def define_module(self):
#         ngf = self.gf_dim
#         # TEXT.DIMENSION -> GAN.CONDITION_DIM
#         self.ca_net = CA_NET()
#         # --> 4ngf x 16 x 16
#         self.encoder = nn.Sequential(
#             conv3x3(3, ngf),
#             nn.ReLU(True),
#             nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True))
#         self.hr_joint = nn.Sequential(
#             conv3x3(self.ef_dim + ngf * 4, ngf * 4),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True))
#         self.residual = self._make_layer(ResBlock, ngf * 4)
#         # --> 2ngf x 32 x 32
#         self.upsample1 = upBlock(ngf * 4, ngf * 2)
#         # --> ngf x 64 x 64
#         self.upsample2 = upBlock(ngf * 2, ngf)
#         # --> ngf // 2 x 128 x 128
#         self.upsample3 = upBlock(ngf, ngf // 2)
#         # --> ngf // 4 x 256 x 256
#         self.upsample4 = upBlock(ngf // 2, ngf // 4)
#         # --> 3 x 256 x 256
#         self.img = nn.Sequential(
#             conv3x3(ngf // 4, 3),
#             nn.Tanh())

#     def forward(self, text_embedding, noise):
#         _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise)
#         stage1_img = stage1_img.detach()
#         encoded_img = self.encoder(stage1_img)

#         c_code, mu, logvar = self.ca_net(text_embedding)
#         c_code = c_code.view(-1, self.ef_dim, 1, 1)
#         c_code = c_code.repeat(1, 1, 16, 16)
#         i_c_code = torch.cat([encoded_img, c_code], 1)
#         h_code = self.hr_joint(i_c_code)
#         h_code = self.residual(h_code)

#         h_code = self.upsample1(h_code)
#         h_code = self.upsample2(h_code)
#         h_code = self.upsample3(h_code)
#         h_code = self.upsample4(h_code)

#         fake_img = self.img(h_code)
#         return stage1_img, fake_img, mu, logvar


# class STAGE2_D(nn.Module):
#     def __init__(self):
#         super(STAGE2_D, self).__init__()
#         self.df_dim = cfg.GAN.DF_DIM
#         self.ef_dim = cfg.GAN.CONDITION_DIM
#         self.define_module()

#     def define_module(self):
#         ndf, nef = self.df_dim, self.ef_dim
#         self.encode_img = nn.Sequential(
#             nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
#             nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
#             nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 32),
#             nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
#             conv3x3(ndf * 32, ndf * 16),
#             nn.BatchNorm2d(ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
#             conv3x3(ndf * 16, ndf * 8),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 8
#         )

#         self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
#         self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

#     def forward(self, image):
#         img_embedding = self.encode_img(image)

#         return img_embedding
