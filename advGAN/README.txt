------------README------------
command: python GANTorch.py --dataset_train 'path_to train set' --dataset_test 'path to test set' --model_path 'path to store model'

 1. file structure
    advGAN/
    advGAN/GANTorch.py          --> main process and training
    advGAN/model.py             --> generator, discriminator, DeepSpeech models
    advGAN/dataloader.py        --> audio data loader for training and testing dataset

2.  parser arguments
    epoch                       --> start from which epoch? 0 if from scratch       (default = 0)                
    n_epoch                     --> train for how many epochs?                      (default = 200)
    dataset_name                --> name of dataset                                 (default = 'TIMIT')
    dataset_train               --> path of the training dataset                    (default = none)
    dataset_test                --> path of the training dataset                    (default = none)
    batch_size                  --> size of batch_size                              (default = 128)
    lr                          --> learning rate (adam)                            (default = 0.0001)
    b1                          --> decay of first order momentum of gradient       (default = 0.5)
    b2                          --> same as above (for adam optimizer)              (default = 0.999)
    decay_epoch                 --> epoch to start learning rate decay              (default = 100)
    n_cpu                       --> number of cpu threads to use                    (default = 4)
    sample_interval             --> interval to create generated adversarial example(default = 10)
    checkpoint_interval         --> interval to save checkpoint models              (default = 50)
    n_residual_blocks           --> number of residual blocks in generator          (default = 4)
    bound_noise                 --> user-specified bound of perturbation magnitude  (default = 0.05)
    model_path                  --> path to target model, ex: DeepSpeech            (default = 'deepspeech_final.pth')

    *** MUST GIVE PARAMS ***
    dataset_train, dataset_test, model_path
    
    *** experiment params ***
    n_epoch, batch_size, decay_epoch, bound_noise, lr

    *** increase/decrease output adversarial examples ***
    sample_interval, checkpoint_interval

    *** don't change ***
    b1, b2, n_cpu, n_residual_blocks

3. results
    - generated adversarial examples will be under 'audios/TIMIT/', the filename indicating the number of batches done
    - checkpoint models will be under 'saved_models/TIMIT/', G and D respectively for generator and discriminator