# model
import numpy as np
import tensorflow as tf

class Model():
    def __init__(self):
        self.batch_size = None
        self.lstm_hidden = None
    
    def extractor(self, data_batch):
        # create a CNN feature extractor
        # input: 1x400 (frame size) -> reshape(1x20x20) -> conv1 -> conv2 -> conv3 -> reshape(64x100)
        def conv_block(X, f, k, s):
            conv = tf.layers.conv2d(inputs=X, filters=f, kernel_size=k, strides=s, padding='same')
            norm = tf.layers.batch_normalization(conv) 
            out  = tf.nn.relu(norm)
            return out
        print('Shape of utterance batch: ', data_batch.shape)
        data = tf.reshape(data_batch, [-1, 1, 20, 20])
        data = tf.transpose(data, [0,3,2,1])
        print('Output size of reshape: ', data.shape)
        c1 = conv_block(data, 16, 7, 1)                         
        print('Output size of conv1: ', c1.shape)
        c2 = conv_block(c1, 32, 5, 1)                           
        print('Output size of conv2: ', c2.shape)
        c3 = conv_block(c2, 64, 3, 2)                           
        print('Output size of conv3: ', c3.shape)
        out = tf.reshape(c3, [-1, 100, 64])
        out = tf.transpose(out, [0, 2, 1])        
        print('Output size of extractor: ', out.shape)
        
        return out

    def lstm_seq(self, X):
        # create a lstm sequence trainer
        rnn = tf.contrib.rnn.LSTMBlockCell(num_units=128)
        print('Dim of lstm input: ', X.shape)
        num_frames = X.shape[0]
        state = rnn.zero_state(num_frames, tf.float32)
        (output, newstate) = tf.nn.dynamic_rnn(cell=rnn, inputs=X, initial_state=state)
        print('Output size of lstm seq: ', output.shape)
        output = tf.reduce_mean(output,2)   #reduce along 128(timestep)
        print('Output size of mean seq: ', output.shape)
        return output


    def classifier(self, X):
        # create a DNN classifier
        # input: 64-256 FC1 -> 256-256 FC2 -> 256-2 linear
        def fc_layer(X):
            dense = tf.layers.dense(inputs=X, units=256)
            norm  = tf.layers.batch_normalization(dense)#, training=True)
            out   = tf.nn.relu(norm)
            return out
        
        print('input to DNN: ', X.shape)
        fc1 = fc_layer(X)
        fc2 = fc_layer(fc1)
        out = tf.layers.dense(inputs=fc2, units=2)
        #out = tf.reduce_mean(out, 0)
        #out = tf.transpose(out)
        print('output of DNN: ', out.shape)
        return out