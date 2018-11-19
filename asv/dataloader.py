# Dataloader
import io
import os
import time
import numpy as np
import librosa
import tensorflow as tf

class Dataloader():
    def __init__(self, file_dir=None, speaker_list=None, f_param=None, thresh=None):
        self.file_dir = file_dir
        self.speakers = speaker_list
        self.sr = f_param[0]
        self.frame_size = f_param[1]
        self.frame_shift = f_param[2]
        self.batch_size = None
        self.threshold = thresh
        self.waveset = []
        self.labelset = []
        self.real_count = 0
        self.spoof_count = 0
        
    def create_dataset(self):
        assert self.file_dir, "No directory given!"
        assert self.speakers, "No speaker list given!"
        print(self.file_dir)
        print(self.speakers)
        try:
            with io.open(self.speakers, 'r', encoding='utf-8') as speaker_file:
                print("Creating dataset!")
                start_time = time.time()
                speakers = speaker_file.readlines()
            for groundtruth in speakers:
                #parse line into folder / filename / type(human or Sx) / (human or spoof)
                tokens = groundtruth.split(' ')
                wav_path = self.file_dir + '/' + tokens[0] + '/' + tokens[1] + '.wav'
                
                if tokens[3].strip() == 'human' and self.real_count > 0.3*self.threshold:
                    continue

                try:
                    data, sample_r = librosa.load(wav_path, offset=0.25 ,duration=3.0, sr=16000)
                    # discard if utterance < 3s
                    if librosa.get_duration(y=data, sr=sample_r) < 3:
                        #print('Discarded audio, length < 3s...')
                        continue
                    #print('wav sampled at {} Hz, \ndata: {}'.format(sample_r, data[1000:1010]))
                except:
                    #print('Error loading file... %s not found' %(tokens[1]))
                    pass
                else:
                    try:
                        #print("Data size: ", data.size)
                        batch_size = int((data.size - self.frame_size) / self.frame_shift)
                        if not self.batch_size: 
                            self.batch_size = batch_size
                        assert self.batch_size == batch_size, "Batch size mismatch!"
                        #print('Num of samples in utterance: %i' % batch_size)
                        batch_set = np.array([data[self.frame_shift*i : self.frame_shift*i+self.frame_size] for i in range(batch_size)]) #BUG!
                        # should be [batch_size x 1 x 400] not [batch_size x 400]
                        #print("Creating dataset... size of utterance set: ", batch_set.size) 
                        self.waveset.append(batch_set)
                        if tokens[3].strip() == 'human':
                            blah = np.array([1 for _ in range(batch_size)])
                            self.labelset.append(blah)
                            self.real_count += 1
                            #print('real_count:', self.real_count)
                        else:
                            blaj = np.array([0 for _ in range(batch_size)])
                            self.labelset.append(blaj)
                            self.spoof_count += 1
                            #print('spoof_count:', self.spoof_count)
                        ##################################################################
                        if len(self.waveset)>self.threshold: break                        

                    except Exception as e:
                        print(e)
            
            self.waveset = np.array(self.waveset)
            self.labelset = np.array(self.labelset)
            end_time = time.time()
            print('Finished creating dataset, size of dataset:', len(self.waveset))
            print("Time elapsed: {}".format(end_time-start_time))
            print('Human: {} Spoof: {}'.format(self.real_count, self.spoof_count))

        except:
            print("No input directory given!")

    def shuffle(self):
        assert self.waveset is not None, "Waveset not created yet!"
        assert self.labelset is not None, "Labelset not created yet!"
        # BUG: dim mismatch !!
        print ('Shuffling!')
        #print ('Shape of waveset: {}'.format(self.waveset.shape))
        #print ('Shape of labelset: {}'.format(self.labelset.shape))
        s = self.set_size()
        temp = np.concatenate((self.waveset, np.reshape(self.labelset, (s[0],s[1],1))), axis=2)
        np.random.shuffle(temp)
        # self.waveset, self.labelset = np.split(temp, [400,401], axis=2)
        result = np.split(temp, [400,401], axis=2)
        self.waveset = result[0]
        self.labelset = np.reshape(result[1], (s[0],s[1]))
        print ('Shuffled!')
        #print ('Shape of waveset: {}'.format(self.waveset.shape))
        #print ('Shape of labelset: {}'.format(self.labelset.shape))

    
    def get_batch(self, index):
        try:
            return (self.waveset[index], self.labelset[index])
        except IndexError as e:
            print(e)
    
    def set_size(self):
        #assert len(self.waveset) == len(self.labelset), "Mismatch of wav and label dataset!"
        #return len(self.waveset)
        assert self.waveset.shape[0] == self.labelset.shape[0], "Mismatch of wav and label dataset!"
        return self.waveset.shape

    def get_count(self):
        assert self.spoof_count, "dataset error"
        assert self.real_count, "dataset error"
        return self.real_count, self.spoof_count
    
    def sample_size(self):
        return self.set_size()[0]



if __name__ == '__main__':
    curpath = os.getcwd()
    fps = (16000,400,160)
    testloader = Dataloader(file_dir=os.path.join(curpath,'wav'), speaker_list=os.path.join(curpath, 'CM_protocol/cm_develop.ndx'), f_param=fps)
    testloader.create_dataset()
    testloader.shuffle()