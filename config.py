# !/usr/bin/python3
config = {
    #common
    'base_folder': "model",             #folder for models, weights, PNGs etc
    'gen_weights': "gen_weights.h5",    #Weights of generator
    'dis_weights': "dis_weights.h5",    #Weights of discriminator
    'gen_model': "gen.yaml",            #Description of generator
    'dis_model': "dis.yaml",            #Description of discriminator
    'gen_picture': 'gen.png',           #Image of generator
    'dis_picture': 'dis.png',           #Image of discriminator

    #tensorboard
    'tensorflow_logs':
        '/home/kell/tensorflow_logs',   #Place where logs are stored (for tensorboard)

    #train
    'epochs': 2,                        #Number of epochs to train
    'opt_lr': 0.1,

    #test
    'test_seq': [   #Sequence for testing
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],   #C
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   #Am
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],   #F
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   #G
    ],
    'test_freq': 8,                     #How often to insert data from test_seq
    'test_length': 32,                  #Length of resulting sequence
    'result_file': 'result.txt',        #Location of resulting file
}