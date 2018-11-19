# An Investigation of Deep-Learning Frameworks for Speaker Verification Antispoofing #
from __future__ import print_function
import os
import io
import time
import numpy as np
import tensorflow as tf
from config import get_config
from dataloader import Dataloader 
from model import Model
    
class ASV():
    def __init__(self, params):
        self.datapath = params.data_path if params.data_path else os.getcwd()
        self.checkpoint_path = params.checkpoint_path 
        self.log_path = params.log_path 
        self.learning_rate = params.learning_rate 
        self.epochs = params.epoch 
        self.early_stop = params.early_stop 
        self.sr = params.sr
        self.frame_size = params.frame_size 
        self.frame_shift = params.frame_shift 
        # create datasets
        self.trainset = None 
        self.devset = None 
        self.testset = None 
        # create model
        self.model = Model()
    

    def create_dataset(self, mode):
        # set directory for different modes and create respective datasets
        if mode == 'train':
            s_list = os.path.join(self.datapath, 'CM_protocol/cm_train.trn')
            t = 15000
        elif mode == 'dev':
            s_list = os.path.join(self.datapath, 'CM_protocol/cm_develop.ndx')
            t = 10000
        elif mode == 'test':
            s_list = os.path.join(self.datapath, 'CM_protocol/cm_evaluation.ndx')
            t = 10000
        else:
            print('Invalid mode! specify one of train, dev, test.')
        f = (self.sr, self.frame_size, self.frame_shift)
        DLoad = Dataloader(file_dir=os.path.join(self.datapath,'wav'), speaker_list=s_list, f_param=f, thresh=t)
        DLoad.create_dataset()
        return DLoad

    def train(self):
        log_file = open('training_model_shuffle_test.txt','w')
        log_file.write("Training model....\n")
        print('========= Training model =========')

        #self.trainset = training_model.create_dataset('train')
        self.trainset = self.create_dataset('test')
        assert self.trainset, "No training data given!"

        tf.reset_default_graph()
        # utterance is a preprocesses 3D tensor with size [num_frames x 1 x 400]
        utterance = tf.placeholder(shape=(self.trainset.batch_size, self.frame_size), dtype=tf.float32)
        utter_label = tf.placeholder(shape=(self.trainset.batch_size), dtype=tf.int32)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # create graph
        extracted_features = self.model.extractor(utterance)
        sequence = self.model.lstm_seq(extracted_features)
        classification = self.model.classifier(sequence)

        # define loss sigmoid cross entropy loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=utter_label, logits=classification, name='train_loss'))

        # update loss
        trainable_vars = tf.trainable_variables()                                           # get variable list
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)                # get optimizer 
        grads, vars = zip(*optimizer.compute_gradients(loss))                               # compute gradients of variables with respect to loss
        train_op = optimizer.apply_gradients(zip(grads, vars), global_step=global_step)     # gradient update operation

        # check variables memory
        variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
        log_file.write("total variables :{}\n".format(variable_count))

        # record loss
        loss_summary = tf.summary.scalar("train_loss", loss)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(var_list=trainable_vars)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            if os.access(self.checkpoint_path, os.F_OK):
                for file in os.listdir(self.checkpoint_path): 
                    os.remove(os.path.join(self.checkpoint_path, file)) 
            else:
                os.makedirs(self.checkpoint_path) 
            
            if not os.access(self.log_path, os.F_OK):
                os.makedirs(self.log_path)
            writer = tf.summary.FileWriter(self.log_path, sess.graph)
            
            best = [0,0]
            stopped = False
            print('Training....')
            start_time = time.time()
            for epoch in range(self.epochs):
                total_loss = 0
                self.trainset.shuffle()
                
                for i in range(self.trainset.sample_size()):
                    (audio, label) = self.trainset.get_batch(i)
                    #log_file.write("Check input shape of utter and labels: {},{}".format(audio.shape, label.shape))
                    cur_loss, _ , summary = sess.run([loss, train_op, merged],feed_dict={utterance:audio, utter_label:label})
                    total_loss += cur_loss
                    log_file.write("Training on epoch #{} utterance #{}, loss:{}\n".format(epoch, i, cur_loss))
                    if self.early_stop:
                        if i == 0:
                            best = [0, cur_loss]
                        elif best[1] > cur_loss:
                            best = [i, cur_loss]
                        else:
                            if i-best[0] > 200:
                                log_file.write('Stopping early! Loss stuck at {}\n'.format(best[1]))
                                stopped = True
                                break

                    if i%15 == 0:
                        writer.add_summary(summary, i)
                
                if stopped: break

                saver.save(sess, os.path.join(self.checkpoint_path, "model%i.ckpt"%(epoch)))

                end_time = time.time()
                log_file.write("Epoch:{}     Training loss:{}\n".format(epoch, total_loss/self.trainset.sample_size()))
                log_file.write("Time elapsed: {} min\n".format((end_time-start_time)/60))
                print("Epoch:{}     Training loss:{}\n".format(epoch, total_loss/self.trainset.sample_size()))
                print("Time elapsed: {} min".format((end_time-start_time)/60))
            
            log_file.write('Finished training. Best loss: {}\n'.format(best[1]))
            log_file.close()
            print('Finished training. Best loss: {}'.format(best[1]))


    def dev(self):
        # cross validation with dev set
        pass

    def test(self, num_model):
        log_file = open('testing_model_shuffle_test_{}.txt'.format(num_model),'w')
        log_file.write("Testing model #{}....\n".format(num_model))
        print('========= Testing model =========')

        self.testset = self.create_dataset('dev')
        assert self.testset, "No testing data given!"
        human_count, spoof_count = self.testset.get_count()

        tf.reset_default_graph()
        # utterance is a preprocesses 3D tensor with size [num_frames x 1 x 400]
        utterance = tf.placeholder(shape=(self.testset.batch_size, self.frame_size), dtype=tf.float32)
        
        # create graph
        extracted_features = self.model.extractor(utterance)
        sequence = self.model.lstm_seq(extracted_features)
        classification = self.model.classifier(sequence)


        saver = tf.train.Saver(var_list=tf.global_variables())

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            # load model
            print('Loading model#{}'.format(num_model))
            log_file.write('model path: {}\n'.format(self.checkpoint_path))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path)
            ckpt_list = ckpt.all_model_checkpoint_paths
            for model in ckpt_list:
                name = "model{}.ckpt".format(num_model)
                if name == model:
                    log_file.write('loading model: {}\n'.format(model))
                    saver.restore(sess, model)
                    break
            
            print('Testing....')
            start_time = time.time()
            true_human = true_spoof = false_human = false_spoof = 0
            for i in range(self.testset.sample_size()):
                (audio, label) = self.testset.get_batch(i)

                classified = sess.run(classification,feed_dict={utterance:audio})
                #print("classification: ", classified)#.shape)
                tf.reduce_mean(classified, 0)
                #print("shape of reduced classification: ", classified.shape)
                #print('results: ',classified)
                infer_label = 1 if classified[0][0] < classified[1][0] else 0
                log_file.write('infer label: {}\n'.format(infer_label))
                # use precision to count true_positive / false_positive / true_negative / false_negative
                if infer_label == 1 and label[0] == 1: 
                    true_human += 1
                    log_file.write('index {} correct: human!\n'.format(i))
                elif infer_label == 1 and label[0] == 0:
                    false_human += 1
                    log_file.write('index {} wrong: human!\n'.format(i))
                elif infer_label == 0 and label[0] == 0:
                    true_spoof += 1
                    log_file.write('index {} correct: spoof!\n'.format(i))
                elif infer_label == 0 and label[0] == 1:
                    false_spoof += 1 
                    log_file.write('index {} wrong: spoof!\n'.format(i))
                else:
                    log_file.write('Invalid infer case!\n')
            
            end_time = time.time()
            total = human_count+spoof_count
            log_file.write('Results for model #{}\n'.format(num_model))
            log_file.write('Total testing utterances: #{}\n#human: {}\n#spoof: {}\n'.format(total, human_count, spoof_count))
            log_file.write('True positive {}%\n'.format(100 * true_human/human_count))     #true_pos = true_human/total_human
            log_file.write('True negative {}%\n'.format(100 * true_spoof/spoof_count))     #true_neg = true_spoof/total_spoof
            log_file.write('False positive {}%\n'.format(100 * false_human/spoof_count))   #false_pos = false_human/total_spoof
            log_file.write('False negative {}%\n'.format(100 * false_spoof/human_count))   #false_neg = false_spoof/total_human
            log_file.write("Time elapsed: {}".format(end_time-start_time))
            print('Finished testing! Results for model #{}: true negative: {}'.format(num_model, true_spoof/spoof_count))
            print("Time elapsed: {} min".format((end_time-start_time)/60))
        
        log_file.close()


if __name__ == '__main__':
    params = get_config()
    training_model = ASV(params)
    #training_model.create_dataset('train')
    training_model.train()
    for index in range(params.epoch):
        training_model.test(index)
