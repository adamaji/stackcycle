from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils import model_zoo
from torchvision import models
import scipy

from miscc.datasets import TextDataset
from miscc.config import cfg

from model import STAGE1_G, STAGE1_D, STAGE1_ImageEncoder

import pdb

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

DATA_DIR = "../data/flowers"
cfg.TEXT.DIMENSION = 300
cfg.GAN.CONDITION_DIM = 128
cfg.GAN.DF_DIM = 96
cfg.GAN.GF_DIM = 192

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)    
    
image_transform = transforms.Compose([
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = TextDataset(DATA_DIR, 'train',
		      imsize=64,
		      transform=image_transform)
assert train_dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    drop_last=True, shuffle=True)

######

image_transform = transforms.Compose([
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = TextDataset(DATA_DIR, 'test',
		      imsize=64,
		      transform=image_transform)
assert test_dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size ,
    drop_last=True, shuffle=True)

#imsize = 784
imsize = 64 * 64 * 3
#imsize = 16 * 16 * 3
zsize = 300

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(imsize, 400)
        self.fc21 = nn.Linear(400, 50)
        self.fc22 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 400)
        self.fc4 = nn.Linear(400, imsize)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.netG = STAGE1_G()
        # state_dict = \
        #     torch.load("../output/birds_stageI_2018_03_19_15_55_52/Model/netG_epoch_120.pth",
        #                map_location=lambda storage, loc: storage)
        # self.netG.load_state_dict(state_dict)
        # print('Load from: ', cfg.NET_G)        
        
        #self.image_encoder = CNN_ENCODER(128)
        
        self.image_encoder = STAGE1_ImageEncoder()
        # state_dict = \
        #     torch.load("../output/birds_stageI_2018_03_19_15_55_52/Model/image_encoder_epoch_120.pth",
        #                map_location=lambda storage, loc: storage)
        # self.image_encoder.load_state_dict(state_dict)
        
        
        ndf, nef = 60, 128
        
        self.nef = nef
        
        self.decode_lin = nn.Sequential(
            nn.Linear(zsize, nef * 4 * 4),
            nn.BatchNorm1d(nef * 4 * 4),
            nn.ReLU(True)
        )
        
        self.decode_img = nn.Sequential(
            
            nn.ConvTranspose2d(nef, nef // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nef // 2, nef // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nef // 4, nef // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(nef // 8, nef // 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nef // 16, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
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
            #nn.MaxPool2d(2, stride=2),
            #nn.Linear(1024, 300)
        )
        self.l1 = nn.Linear(480 * 4 * 4, zsize)
        self.l2 = nn.Linear(480 * 4 * 4, zsize)
        
        self.l = nn.Linear(128, zsize * 2)
        self.relu = nn.ReLU()
        #######     

    def encode(self, x):
        # h1 = self.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        #hidden = self.encode_img(x)
        
        encoded = self.image_encoder(x)
        fake_img_feats, fake_img_emb, fake_img_code, mu, logvar = encoded
        
        #_, encoded = self.image_encoder(x)
                
        #line = self.relu(self.l(encoded))
        #mu = line[:,:300]
        #logvar = line[:,300:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        #if self.training:
        if True:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # h3 = self.relu(self.fc3(z))
        # return self.sigmoid(self.fc4(h3)) 
        _, fake_img, _, _ = self.netG(z, None)
        img = self.sigmoid(fake_img)
        return img

    def forward(self, x):
        mu, logvar = self.encode(x)#.view(-1, imsize))
        
        z = self.reparameterize(mu, logvar)
                
        #z_p = self.decode_lin(z).view(-1, self.nef, 4, 4)
        #img = self.decode_img(z_p)

        #z = self.encode_img(x).view(-1, 480 * 4 * 4)
        img = self.decode(z)
        # mu = None
        # logvar = None
        
        return img, mu, logvar

    
model = VAE()

netD = STAGE1_D()

if args.cuda:
    model.cuda()
    netD.cuda()
    
model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_params, lr=1e-3)
optd = optim.Adam(netD.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #pdb.set_trace()
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.smooth_l1_loss(recon_x, F.sigmoid(x), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)    
    
    return BCE + KLD, BCE, KLD
    #return BCE, BCE, KLD


def train(epoch):
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        _, real_img_cpu, _, _, _ = data
        data = real_img_cpu
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        
#         #real_feats_g = netD(data)
#         #real_logits_g = netD.get_uncond_logits(real_feats_g)
#         fake_feats_g = netD(recon_batch)
#         fake_logits_g = netD.get_uncond_logits(fake_feats_g)
#         real_labels_g = Variable(torch.FloatTensor(args.batch_size).fill_(0.9)).cuda()
#         #fake_labels_g  = Variable(torch.FloatTensor(args.batch_size).fill_(0)).cuda()
        
#         loss_fake_g = criterion(F.sigmoid(fake_logits_g), real_labels_g)
#         #loss_real_g = criterion(F.sigmoid(real_logits_g), fake_labels_g)
        
#         loss_g = loss_fake_g
        
#         loss = loss + loss_g
        
        loss.backward()
        
        train_loss += loss.data[0]
        optimizer.step()
        
#         optd.zero_grad()
        
#         real_feats = netD(data.detach())
#         real_logits = netD.get_uncond_logits(real_feats)
#         fake_feats = netD(recon_batch.detach())
#         fake_logits = netD.get_uncond_logits(fake_feats)
#         real_labels = Variable(torch.FloatTensor(args.batch_size).fill_(0.9)).cuda()
#         fake_labels = Variable(torch.FloatTensor(args.batch_size).fill_(0.1)).cuda()
        
#         loss_fake = criterion(F.sigmoid(fake_logits), fake_labels)
#         loss_real = criterion(F.sigmoid(real_logits), real_labels)
        
#         d_loss = loss_fake + loss_real
#         d_loss.backward()  
        
#         optd.step()
        
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} {:.4f} {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data),
                bce.data[0] / len(data),
                kld.data[0] / len(data)
            ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        _, real_img_cpu, _, _, _ = data
        data = real_img_cpu
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        loss, _, _ = loss_function(recon_batch, data, mu, logvar)
        test_loss += loss.data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([F.sigmoid(data[:n]),
                                  recon_batch.view(args.batch_size, 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                     'vae_results/reconstruction_' + str(epoch) + '.png', nrow=n, normalize=True)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    #test(epoch)
    train(epoch)
    if epoch % 1 == 0:
        test(epoch)
        sample = Variable(torch.randn(64, zsize))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu().view(64, 3, 64, 64)
        save_image(sample.data,
                   'vae_results/sample_' + str(epoch) + '.png', normalize=True)
