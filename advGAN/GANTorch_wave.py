import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable
from model_wave import *
from decoder import GreedyDecoder
from dataloader_wave import *

import torch.nn as nn
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
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=500, help='interval between saving model checkpoints')
parser.add_argument('--bound_noise', type=float, default=0.05, help='user-specified bound of perturbation magnitude')
parser.add_argument('--model_path', type=str, default='', help='path to the target ASR model')
parser.add_argument('--output_path', type=str, default='/mnt/advGAN/', help='output path')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('%s/audio/%s' % (opt.output_path, opt.dataset_name), exist_ok=True)
os.makedirs('%s/saved_models/%s' % (opt.output_path, opt.dataset_name), exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_Adv = torch.nn.MSELoss()
criterion_hinge = lambda a,b: max(0, a - b) 

cuda = True if torch.cuda.is_available() else False

# Training data loader  
model = DeepSpeech.load_model(opt.model_path, cuda=cuda)
labels = DeepSpeech.get_labels(model)
audio_conf = DeepSpeech.get_audio_conf(model)
decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

train_dataset = audioDataset(filepath=opt.dataset_train, audio_conf=audio_conf)
#train_dataloader = audioLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu)

test_dataset = audioDataset(filepath=opt.dataset_test, audio_conf=audio_conf)
#test_dataloader = audioLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, shuffle=True)

G = Generator()         # give i/o dimensions: input = 2**16 = [65536], output = 2**16 = [65536]
D = Discriminator()     # give i/o dimensions: input = 2**16 = [65536], output = [batch_size, 1024]

if cuda:
    G = G.cuda()
    D = D.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load('%s/saved_models/%s/G_%d.pth' % (opt.output_path, opt.dataset_name, opt.epoch)))
    D.load_state_dict(torch.load('%s/saved_models/%s/D_%d.pth' % (opt.output_path, opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

# Loss weights      --> can be adjusted
lambda_gan = 0.9
lambda_hinge = 1 - lambda_gan

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# audio functions
def reconstruct(audio, path, rate):
    audio = audio.transpose(0,1)
    audio_tensor = audio.data.squeeze()
    audio_np = audio_tensor.cpu().numpy()
    audio_np = np.round(audio_np)
    audio_np = np.clip(audio_np, -2**15, 2**15-1)
    audio_np = np.array(audio_np,dtype=np.int16)
    wav.write(path, rate, audio_np)


def sample_audios(test_ds,epoch,index, rate): #batches_done
    #wavs, trans= next(test_ds_iter)
    wavs, trans = test_ds[index]
    real = Variable(wavs, requires_grad=False).cuda()          #BUGGY LINE
    fake = G(real)
    fake_data = fake + real
    #path = '%s/audio/%s/test_%d_%d.wav' % (opt.output_path, opt.dataset_name, epoch, batches_done)
    path = '%s/audio/%s/test_%d_%d.wav' % (opt.output_path, opt.dataset_name, epoch, index)
    reconstruct(fake_data, path, rate)
    return fake_data, trans
   

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    #test_set_iter = iter(test_dataloader)
    for i in range(len(train_dataset)):
    #for i, (batch) in enumerate(train_dataloader):
        audio,transcript = train_dataset[i]
        #audio,transcript = batch
        
        #batch_size = len(audio)
        batch_size = opt.batch_size

        real_data = Variable(audio, requires_grad=False).cuda()
        # Adversarial ground truths       
        valid = Variable(Tensor(np.ones((batch_size, 2**10))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((batch_size, 2**10))), requires_grad=False)
        #print('valid:',valid.size())

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        # GAN loss
        fake_noise = G(real_data)  
        fake_noise = Variable(fake_noise, requires_grad=True)         
        fake_data = fake_noise + real_data
        #print('fake_data',fake_data.size()) 
        loss_GAN = criterion_GAN(D(fake_data), valid) 

        # Adv loss
        real_spec = get_spectrogram(real_data,audio_conf)
        #print("real_spec:", real_spec.size())
        out_real = model(real_spec)
        model_real = Variable(out_real, requires_grad=False)
        out_real = out_real.transpose(0,1)
        orig_trans = decoder.decode(out_real.data)

        fake_spec = get_spectrogram(fake_data,audio_conf)
        out_fake = model(fake_spec)
        model_fake = Variable(out_fake, requires_grad=False)
        out_fake = out_fake.transpose(0,1)
        generated_trans = decoder.decode(out_fake.data)
        loss_Adv = criterion_Adv(model_real, model_fake)

        # hinge loss
        # if loss is too large, try using (fake_noise - real_data)
        #fake_noise = Variable(fake_data-real_data, requires_grad=False)
        norm_noise = torch.norm(fake_noise, 2, 1)
        norm_noise = torch.norm(norm_noise)
        loss_hinge = criterion_hinge(norm_noise, opt.bound_noise)

        loss_G = (loss_Adv + lambda_gan*loss_GAN + lambda_hinge*loss_hinge)
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator
        # -----------------------
        optimizer_D.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D(real_data),valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = criterion_GAN(D(fake_data.detach()),fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        #batches_done = epoch * len(train_dataloader) + i
        batches_done = epoch * len(train_dataset) + i
        #batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        batches_left = opt.n_epochs * len(train_dataset) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if batches_done % opt.checkpoint_interval == 0:
            transcript = ''.join(transcript)
            orig_trans = ''.join(''.join(ch) for ch in orig_trans[0])
            gen_trans = ''.join(''.join(ch) for ch in generated_trans[0])
            sys.stdout.write("Training.... \n[TIMIT transcript:%s]\n[Original transcript:%s] \n[Generated transcript:%s]\n" %(transcript, orig_trans, gen_trans))
        
        sys.stdout.write("[Epoch %d/%d] [Batch %d/%d] \n[D loss: %f] [G loss: %f, gan: %f, adv: %f, hinge: %f] ETA: %s\n" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(train_dataset), #dataloader
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_Adv.item(),
                                                        loss_hinge.item(), time_left))

        # TODO: If at sample interval save audio and calculate error rate
        if batches_done % opt.sample_interval == 0:
            path = '%s/audio/%s/train_%d_%d.wav' % (opt.output_path, opt.dataset_name, epoch, batches_done)
            rate = audio_conf['sample_rate']
            reconstruct(fake_data, path, rate)
        if batches_done % opt.sample_interval == 0:
            test_out, test_trans = sample_audios(test_dataset, epoch, int(batches_done / opt.sample_interval), rate)
            test_out = get_spectrogram(test_out, audio_conf)
            test_out = model(test_out)
            test_out = test_out.transpose(0,1)
            test_gen = decoder.decode(test_out.data)
            #cer = decoder.cer(test_trans[0], test_gen[0])
            #wer = decoder.wer(test_trans[0], test_gen[0])
            test_trans = ''.join(test_trans)
            test_gen = ''.join(''.join(ch) for ch in test_gen[0])
            sys.stdout.write('Testing....\n [TIMIT transcript: %s]\n[Generated transcript:%s]\n' %(test_trans, test_gen))
            

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G.state_dict(), '%s/saved_models/%s/G_%d.pth' % (opt.output_path, opt.dataset_name, epoch))
        torch.save(D.state_dict(), '%s/saved_models/%s/D_%d.pth' % (opt.output_path, opt.dataset_name, epoch))

 
