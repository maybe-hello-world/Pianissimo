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
    'epochs': 1,                        #Number of epochs to train
    'd_opt_lr': 0.03,                   #discriminator's optimizator learning rate
    'g_opt_lr': 0.01,                   #generator's

    # 0-C 1-C# 2-D 3-D# 4-E 5-F 6-F# 7-G 8-G# 9-A 10-A# 11-B
    'test_seq': [
        # first few notes feeded from this pattern, then output -> input except for 'marker' chords
        # Key: E min
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   #C5
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],   #D5

    ],

    'test_freq': 8,                     #How often to insert data from test_seq
    'test_length': 32,                  #Length of resulting sequence
    'result_file': 'result.txt',        #Location of resulting file
}