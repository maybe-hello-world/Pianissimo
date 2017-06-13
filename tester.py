# !/usr/bin/python3
from config import *

import tensorflow as tf
import keras.backend as K
import numpy as np

from keras.models import model_from_yaml

def test():
    test_seq = config['test_seq']

    sess = tf.Session()

    # Define session for Keras and set learning flag to true (in order to use dropout even in tesing phase)
    K.set_session(sess)
    K.set_learning_phase(True)

    #load generator model and weights
    with open(config['base_folder'] + "/" + config['gen_model'], "r") as file:
        gen = model_from_yaml(file.read())
    gen.load_weights(config['base_folder'] + "/" + config['gen_weights'])

    #set input for session
    inp = gen.input

    # from tanh (-1, 1) to {0, 1}
    g_out = gen.output
    #g_out = tf.sign(g_out)
    #g_out = tf.add(g_out, tf.constant(1.0))
    #g_out = tf.div(g_out, tf.constant(2.0))

    #prepare data
    sequence = []
    start_seq = np.array(test_seq[0])

    # feed generator with start sequence in order to teach her a bit
    for x in start_seq:
        sess.run(g_out, feed_dict={inp: np.expand_dims(np.expand_dims(x, axis=0), axis=0)})
        sequence.append(x.tolist())

    start_seq = np.array(test_seq[1])

    # feed generator with his own answers
    for i in range(len(test_seq[0]), config['test_length']):
        if i % config['test_freq'] == 0:
            start_seq = test_seq[int(i / config['test_freq']) - 1]
            start_seq = np.array(start_seq)
            sess.run(g_out, feed_dict={inp: np.expand_dims(np.expand_dims(start_seq, axis=0), axis=0)})
        else:
            start_seq = sess.run(g_out, feed_dict={inp: np.expand_dims(np.expand_dims(start_seq, axis=0), axis=0)})[0]
        sequence.append(start_seq.tolist())

    sess.close()

    #save data
    with open(config['base_folder'] + "/" + config['result_file'], "w") as file:
        for chord in sequence:
            x = [i for i, x in enumerate(chord) if x > 0.3]
            file.write(", ".join(map(str,x)))
            file.write('\n')
