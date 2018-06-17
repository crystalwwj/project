import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import math
from collections import OrderedDict

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Generator(nn.Module):
    def __init__(self, input_dim=2**16, output_dim=2**16, kernel_size=25):
        super(Generator,self).__init__()
        dim = 64
        self.init_model = nn.Sequential(nn.Linear(input_dim, 4*4*dim*16))
        self.init_norm = nn.Sequential(nn.BatchNorm1d(dim*16),nn.ReLU())
        # upsampling blocks
        model = [
            nn.ConvTranspose1d(dim*16, dim*8, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*8),
            nn.ReLU(),
            nn.ConvTranspose1d(dim*8, dim*4, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*4),
            nn.ReLU(),
            nn.ConvTranspose1d(dim*4, dim*2, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*2),
            nn.ReLU()
        ]

        # downsampling blocks         not sure if this part is right!
        model += [
            nn.Conv1d(dim*2, dim*4, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim*4, dim*8, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(dim*8, dim*16, kernel_size=kernel_size, stride=4),
            nn.BatchNorm1d(dim*16),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        # output conv layer
        '''model += [
            nn.Linear(dim*16, out_channels, kernel_size=kernel_size, stride=1),
            nn.ReLU() 
        ]'''
        self.model = nn.Sequential(*model)      
        self.end_model = nn.Sequential(nn.Linear(4*4*dim*16,output_dim))

    def forward(self, x):
        dim = 64
        size = x.size()
        #print('input:', size)
        #self.init_model = nn.Sequential(nn.Linear(size[1], 4*4*dim*16))
        x = self.init_model(x)#.cuda()
        x = x.view(size[0],dim*16,16)   ##chg
        #self.init_norm = nn.Sequential(nn.BatchNorm1d(dim*16),nn.ReLU())
        x = self.init_norm(x)
        #x = torch.unsqueeze(x,0)
        x = self.model(x)
        #x = torch.squeeze(x)
        x = x.view(size[0],4*4*dim*16)
        #self.end_model = nn.Sequential(nn.Linear(x.size()[1],1))
        x = self.end_model(x)       #[:,0]
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=2**4, output_dim=1, kernel_size=25):
        super(Discriminator,self).__init__()
        model = [ 
            nn.Conv1d(input_dim, 2**6, kernel_size=kernel_size, stride=4),     #input layer (c8)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(2**6, 2**8, kernel_size=kernel_size, stride=4),              #c16 layer
            nn.BatchNorm1d(2**8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(2**8, 2**10, kernel_size=kernel_size, stride=4),             #c32 layer
            nn.BatchNorm1d(2**10),
            nn.LeakyReLU(negative_slope=0.2),       # output layer
            nn.Linear(57,output_dim),  # reshape/flatten
            nn.ReLU() #nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        size = x.size()
        x = x.view(size[0],2**4,2**12)
        x = self.model(x)
        x = x.view(size[0],-1)
        #x = torch.squeeze(x)
        #print('discriminator output:', x.size())
        return x

##########################################################################################################################

##############################
#        DeepSpeech
##############################

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None,
                 bidirectional=True, context=20):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(rnn_hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        if not self._bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, cuda=False):
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=package.get('bidirectional', True))
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'hidden_layers': model._hidden_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'bidirectional': model._bidirectional
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf

    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "hidden_size": m._hidden_size,
            "hidden_layers": m._hidden_layers,
            "rnn_type": supported_rnns_inv[m._rnn_type]
        }
        return meta
