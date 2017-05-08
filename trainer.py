# !/usr/bin/python3

import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import generator as gn
import discriminator as ds

import math
import random

import numpy as np
import getopt
import sys
from os import listdir
from os.path import isfile, join

def train(inputfolder, epochs):

    ## -----------------------------------------------------------
    ## Dataset preparation
    #

    files = [join(inputfolder, f) for f in listdir(inputfolder) if isfile(join(inputfolder, f))]

    dataset = []
    for file in files:
        f = open(file, 'r')
        dataset.append(np.array([st.rstrip() for st in f.readlines()]))
        f.close()

    dataset = np.array(dataset)
    #
    ## -----------------------------------------------------------

    sess = tf.Session()

    # Define session for Keras and set learning flag to true (batch normalization etc)
    K.set_session(sess)
    K.set_learning_phase(True)

    ## -----------------------------------------------------------
    ## Start stacking GAN
    # Here because we need to call reset_states() from discriminator and generator models
    #

    # input for choosing real/fake data
    if_real = tf.placeholder(tf.bool)

    # input for GAN
    inp = tf.placeholder(tf.float32, shape=(1, 1, 12))

    # Create generator model
    gen = gn.create_generator(inp)

    # Get output tensor and do some reshaping
    g_out = tf.expand_dims(gen.output, axis=1)

    # If on current step we want to feed discriminator with real data than create
    # tensor from inp. else from result of generator
    x = tf.cond(if_real, lambda: inp, lambda: g_out)

    # Create discriminator model
    dis = ds.create_discriminator(x)

    # Again get output tensor
    d_out = dis.output

    # Finish stacking GAN: add loss functions and different updates
    # So my_gan is a function with such signature:
    # my_gan(sess=<session var>, if_real_input=<if_real value>, vector_input=<12-bit vector value>)
    my_gan = GAN(if_real=if_real, inp=inp, d_out=d_out, dis=dis, gen=gen)

    #
    ## -----------------------------------------------------------

    # Create writer for tensorboard
    summary_writer = tf.summary.FileWriter('/home/kell/tensorflow_logs', graph=sess.graph)

    # Initialize vars
    init = tf.global_variables_initializer()
    sess.run(init)

    # Just test vars
    test = np.expand_dims(np.expand_dims(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]), axis=0), axis=0)

    a = my_gan(sess=sess, if_real_input=False, vector_input=test)
    print(a[1])

    # Close session
    summary_writer.close()
    sess.close()

    # for x in dataset:
    #     x = np.expand_dims(np.expand_dims(x, 1), 1)
    #     print(x.shape)
    #     print(x[20])
    ##

    # for i in range(epochs):
    #     np.random.shuffle(dataset)
    #     for song in dataset:

def GAN(if_real, inp, d_out, dis, gen):
    # Calculate discriminator loss
    dloss = - tf.reduce_mean(tf.to_float(if_real) * tf.log(d_out) - (1 - tf.to_float(if_real)) * tf.log(1 - d_out))

    # Calculate generator's loss
    gloss = - tf.reduce_mean(tf.log(1 - d_out))

    # Define optimizer (for learning rate and beta1 see advices in Deep Convolutional GAN pre-print on arXiv)
    opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    # Compute and apply gradients for discriminator
    grad_loss_dis = opt.compute_gradients(dloss, dis.trainable_weights)
    update_dis = opt.apply_gradients(grad_loss_dis)

    # Compute gradients for generator
    # It will be applied only if gen took part in the party
    # (only if we used it's output so if_real == False)
    grad_loss_gen = opt.compute_gradients(gloss, gen.trainable_weights)

    # list: [(gradient, variable),(gradient, variable)...]
    new_grad_loss_gen = [(tf.cond(if_real, lambda: tf.multiply(grad, 0), lambda: grad), var) for grad, var in grad_loss_gen]

    update_gen = opt.apply_gradients(new_grad_loss_gen)

    # We have to update all other tensors like batch_normalization etc
    def other_updates(model):
        input_tensors = []

        # Get other nodes
        nodes = model.inbound_nodes

        # Get all tensors in single list and get updates for all of them
        for i in nodes:
            input_tensors.append(i.input_tensors)
        ans = [model.get_updates_for(i) for i in input_tensors]

        return ans

    update_other = [other_updates(x) for x in [dis, gen]]

    # Collect all updates in one list...
    # generator should be updated in outer space so I include grads instead of update result
    train_step = [update_gen, update_dis, update_other]
    # ...and losses to other
    losses = [gloss, dloss]

    # Create func pointer to feeding step
    def gan_feed(sess, if_real_input, vector_input):
        nonlocal train_step, losses, if_real, inp, opt

        res = sess.run([train_step, losses], feed_dict={
            if_real: if_real_input,
            inp: vector_input,
        })

        return res

    return gan_feed