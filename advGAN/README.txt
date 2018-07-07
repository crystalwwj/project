------------README------------
command: 
python GANTorch_wave.py --dataset_train 'path_to train set' --dataset_test 'path to test set' --model_path 'path to store model'

 1. File structure
    advGAN/
    ## older versions using spectrogram in / spectrogram out ##
    advGAN/GANTorch.py          --> main process and training
    advGAN/model.py             --> generator, discriminator, DeepSpeech models
    advGAN/dataloader.py        --> audio data loader for training and testing dataset
    
    ## current version using audio(.wav) in / audio out ##
    advGAN/GANTorch_wave.py     --> main process and training
    advGAN/model_wave.py        --> generator, discriminator, DeepSpeech models
    advGAN/dataloader_wave.py   --> audio data loader for training and testing dataset
    advGAN/decoder.py		—-> decoder for deepspeech outputs (viewing transcripts)

2.  Parser arguments
    epoch                  --> start from which epoch? 0 if from scratch       (default = 0)                
    n_epoch                --> train for how many epochs?                      (default = 200)
    dataset_name           --> name of dataset                                 (default = 'timit')
    dataset_train          --> path of the training dataset                    (default = 'timit/')
    dataset_test           --> path of the training dataset                    (default = 'timit/')
    batch_size             --> size of batch_size                              (default = 1)
    lr                     --> learning rate (adam)                            (default = 0.0001)
    b1                     --> decay of first order momentum of gradient       (default = 0.5)
    b2                     --> same as above (for adam optimizer)              (default = 0.999)
    decay_epoch            --> epoch to start learning rate decay              (default = 100)
    n_cpu                  --> number of cpu threads to use                    (default = 4)
    sample_interval        --> interval to create generated adversarial example(default = 500)
    checkpoint_interval    --> interval to save checkpoint models              (default = 500)
    bound_noise            --> user-specified bound of perturbation magnitude  (default = 0.05)
    model_path             --> path to target model, ex: DeepSpeech            (default = 'deepspeech_final.pth')

    *** MUST GIVE PARAMS ***
    dataset_train, dataset_test, model_path
    
    *** experiment params ***
    n_epoch, batch_size, decay_epoch, bound_noise, lr

    *** increase/decrease output adversarial examples ***
    sample_interval, checkpoint_interval

    *** don't change ***
    b1, b2, n_cpu

3. Results
    - generated adversarial examples will be under 'audios/TIMIT/', the filename indicating the number of batches done
    - checkpoint models will be under 'saved_models/TIMIT/', G and D respectively for generator and discriminator
