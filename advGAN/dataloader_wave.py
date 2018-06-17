import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

import librosa
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio
import math


def get_spectrogram(wavs, audio_conf):
    window_size = audio_conf['window_size']
    sample_rate = audio_conf['sample_rate']
    window_stride = audio_conf['window_stride']
    n_fft = int(window_size * sample_rate)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    batch_size = len(wavs)
    spects = torch.zeros(batch_size,1,(1+n_fft/2),410) #what the hell is t????? okay I'm hardcoding it.
    for i in range(batch_size):
        wav = wavs[i]
        wav = wav.cpu().data
        wav = wav.numpy()
        spec_model = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spect, phase = librosa.magphase(spec_model)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect).cuda()
        mean = spect.mean()
        std = spect.std()
        spect = (spect - mean) / std
        spect = torch.unsqueeze(spect,0)
        #print('spect size:', spect.size())
        spects[i].copy_(spect)       #hello bug

    return spects

def _collate_fn(batch):
    batch_size = len(batch)
    audios = torch.zeros(batch_size,2**16)
    transcripts = []
    for i in range(batch_size):
        data = batch[i]
        audio = data[0]
        target = data[1]
        audios[i].copy_(audio)
        transcripts.extend(target)
    return audios, transcripts

    
class audioLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(audioLoader,self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class audioDataset(Dataset):
    def __init__(self, filepath, audio_conf):
        self.path = filepath
        # if load timit dataset -> parse file first
        with open(self.path) as file:
            tokens = file.readlines()
        tokens = [x.strip().split(',') for x in tokens]
        self.tokens = tokens
        self.size = len(tokens)
        self.config = audio_conf
        super(audioDataset,self).__init__()
    
    def __getitem__(self, index):
        sample = self.tokens[index]
        audio_path, transcript_path = sample[0], sample[1]
        print('audio:' , audio_path, 'transcript:' , transcript_path)
        audio = self.get_audio(audio_path)
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        return audio, transcript

    def __len__(self):
        return self.size

    def get_audio(self, file_path):
        audio, sample_rate = torchaudio.load(file_path)
        audio = audio.squeeze()
        size = int(audio.size()[0])
        #print('original size: ', size)
        if size > 2**16:
            target = audio[:2**16]
        else:
            padding_left = (2**16 - size)//2
            padding_right = 2**16 - size - padding_left
            target = torch.zeros(2**16)
            target[padding_left:2**16-padding_right] = audio
        #pad = nn.ReplicationPad1d((padding_left,padding_right))
        #audio = pad(audio)
        #print('padded size: ', target.size())
        #audio = np.pad(audio, n_fft//2, mode='reflect')
        #audio = torch.from_numpy(audio)
        #audio = audio.unsqueeze(0)
        return Variable(target)


