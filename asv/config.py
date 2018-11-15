# configurations
import argparse
import numpy as np

parser = argparse.ArgumentParser()

def get_config():
    config, unparsed = parser.parse_known_args()
    return config

# Data Parameters
data_arg = parser.add_argument_group('Data')
#data_arg.add_argument('--train_path', type=str, default='./train', help="train dataset directory")
#data_arg.add_argument('--test_path', type=str, default='./test', help="test dataset directory")
data_arg.add_argument('--sr', type=int, default=16000, help="sampling rate")
data_arg.add_argument('--frame_size', type=int, default=400, help="frame size (ms)")
data_arg.add_argument('--frame_shift', type=int, default=160, help="frame shift (ms)") #??

# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=128, help="hidden state dimension of lstm")
model_arg.add_argument('--restore', type=bool, default=False, help="restore model or not")
model_arg.add_argument('--model_path', type=str, default='./model', help="model directory to save or load")
model_arg.add_argument('--model_num', type=int, default=0, help="number of ckpt file to load")
model_arg.add_argument('--data_path', type=str, default='', help="directory of audio")
model_arg.add_argument('--checkpoint_path', type=str, default='./checkpoint', help="directory to save checkpoints")
model_arg.add_argument('--log_path', type=str, default='./logs', help="directory to save logs")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=bool, default=False, help="train session or not(test session)")
train_arg.add_argument('--loss', type=str, default='softmax', help="loss type (softmax or contrast)")
train_arg.add_argument('--optim', type=str.lower, default='adam', help="optimizer type")
train_arg.add_argument('--learning_rate', type=float, default=4e-3, help="learning rate")
train_arg.add_argument('--beta1', type=float, default=0.5, help="beta1")
train_arg.add_argument('--beta2', type=float, default=0.9, help="beta2")
train_arg.add_argument('--epoch', type=int, default=4, help="max epoch")
train_arg.add_argument('--early_stop', type=bool, default=False, help="use early stopping or not")
train_arg.add_argument('--comment', type=str, default='', help="any comment")

#config = get_config()
#print(config)           # print all the arguments