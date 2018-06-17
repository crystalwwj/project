import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchaudio

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from model import *
#from dataset import *
from dataloader import *
#from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from, 0 if start from scratch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default='timit', help='name of dataset')
parser.add_argument('--dataset_train', type=str, default="timit/", help='path of the training dataset')
parser.add_argument('--dataset_test', type=str, default="timit/", help='path of the testing dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start learning rate decay')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
#parser.add_argument('--img_height', type=int, default=256, help='size of image height')
#parser.add_argument('--img_width', type=int, default=256, help='size of image width')
#parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=10, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=4, help='number of residual blocks in generator')
parser.add_argument('--bound_noise', type=float, default=0.05, help='user-specified bound of perturbation magnitude')
parser.add_argument('--model_path', type=str, default='', help='path to the target ASR model')
parser.add_argument('--output_path', type=str, default='/mnt/advGAN/', help='output path')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('%s/audio/%s' % (opt.output_path, opt.dataset_name), exist_ok=True)
os.makedirs('%s/saved_models/%s' % (opt.output_path, opt.dataset_name), exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss() #lambda a,b: math.log(a) + math.log(1-b)
criterion_Adv = torch.nn.MSELoss()
criterion_hinge = lambda a,b: max(0, a**2 - b) 
#criterion_identity = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, 161, 293)

# Training data loader      TODO!!!  
model = DeepSpeech.load_model(opt.model_path, cuda=cuda)
labels = DeepSpeech.get_labels(model)
audio_conf = DeepSpeech.get_audio_conf(model)
AudioDataset_train = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=opt.dataset_train, labels=labels, normalize=True)
AudioDataset_test = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=opt.dataset_test, labels=labels, normalize=True)
train_dataloader = AudioDataLoader(AudioDataset_train, batch_size=opt.batch_size, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = AudioDataLoader(AudioDataset_test, batch_size=opt.batch_size, num_workers=1)

# Initialize generator and discriminator and model
G = GeneratorResNet(batch_size=opt.batch_size, res_blocks=opt.n_residual_blocks)
D = Discriminator(labels=labels)

if cuda:
    G = G.cuda()
    D = D.cuda()
    # criterion_GAN.cuda()
    # criterion_Adv.cuda()
    # criterion_hinge.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load('%s/saved_models/%s/G_%d.pth' % (opt.output_path, opt.dataset_name, opt.epoch)))
    D.load_state_dict(torch.load('%s/saved_models/%s/D_%d.pth' % (opt.output_path, opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

# Loss weights      --> to be adjusted
lambda_gan = 0.5
lambda_hinge = 1 - lambda_gan

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
'''transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]'''


#def sample_audios(batches_done):            # do i need this?
'''    wavs, t, i, ts = next(iter(val_dataloader))
    real = Variable(wavs.type(Tensor))          #BUGGY LINE
    fake = G(real)
    fake = fake.transpose(0,1)                  #BUGGY LINE
    fake_tensor = fake.data
    output_name = '%s/audio/%s/%d.flac' %(opt.output_path, opt.dataset_name, batches_done)
    torchaudio.save(output_name, fake_tensor, 16000)
'''
# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (batch) in enumerate(train_dataloader):

        inputs, targets, input_percentages, target_sizes = batch
        real_data = Variable(inputs, requires_grad=False).cuda()
        target_sizes = Variable(target_sizes, requires_grad=False)
        #print('inputs:',inputs)
        #print('targets',targets)
        #print(input_percentages)
        #print(target_sizes)
        #targets = Variable(targets, requires_grad=False)
        # Set model input
        #real_data = Variable(batch.type(Tensor))
        #real_labels = model(real_data)
        #print('real_labels',real_labels)
        #print(real_labels.size())

        # Adversarial ground truths             this part is weird
        #valid = Variable(Tensor(np.ones(real_data.size(0), *patch)), requires_grad=False)
        #fake = Variable(Tensor(np.zeros(real_data.size(0), *patch)), requires_grad=False)
        valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
        valid = valid.unsqueeze(0)
        fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)
        fake = fake.unsqueeze(0)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Adv loss
        fake_noise = G(real_data)
        print('real_data:',real_data.size())
        fake_data = fake_noise + real_data       #changed
        print('fake_data',fake_data.size())
        print('valid:',valid.size())
        loss_adv = criterion_Adv(D(fake_data), valid)   #valid dimensions??????? 

        # GAN loss
        model_real = Variable(model(real_data), requires_grad=False)
        #model_real = model_real.transpose(1,2) #squeeze()
        #print(model_real.size())
        model_fake = Variable(model(fake_data), requires_grad=False)
        #model_fake = torch.unbind(model_fake,2)
        #model_fake = Variable(torch.tensor(model_fake, dtype=torch.int32))
        #length = model_real.size(0)
        #sizes = Variable(input_percentages.mul_(int(length)).int(), requires_grad=False)
        #loss_GAN = criterion_GAN(model_real, real_labels, sizes,  target_sizes)
        loss_GAN = criterion_GAN(model_real,model_fake)

        # hinge loss
        #fake_noise = Variable(fake_data-real_data, requires_grad=False)        #changed
        norm_noise = torch.norm(fake_noise,2,2)
        norm_noise = torch.norm(norm_noise,2,1)
        norm_noise = torch.norm(norm_noise)
        loss_hinge = criterion_hinge(norm_noise, opt.bound_noise)

        loss_G = loss_adv + lambda_gan*loss_GAN + lambda_hinge*loss_hinge
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_Adv(D(real_data),valid)
        # Fake loss (on batch of previously generated samples)
        #fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_Adv(D(fake_data.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, gan: %f, adv: %f, hinge: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(train_dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_adv.item(),
                                                        loss_hinge.item(), time_left))

        # TODO: If at sample interval save audio and calculate error rate
        #if batches_done % opt.sample_interval == 0:
        #    sample_audios(batches_done)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), '%s/saved_models/%s/G_%d.pth' % (opt.output_path, opt.dataset_name, epoch))
        torch.save(D.state_dict(), '%s/saved_models/%s/D_%d.pth' % (opt.output_path, opt.dataset_name, epoch))

