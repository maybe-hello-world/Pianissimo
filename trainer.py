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

    ## Load all files from directory
    #

    files = [join(inputfolder, f) for f in listdir(inputfolder) if isfile(join(inputfolder, f))]

    dataset = []
    for file in files:
        f = open(file, 'r')
        dataset.append(np.array([st.rstrip() for st in f.readlines()]))
        f.close()

    dataset = np.array(dataset)
    GAN()
    # for x in dataset:
    #     x = np.expand_dims(np.expand_dims(x, 1), 1)
    #     print(x.shape)
    #     print(x[20])
    ##

    ## Define tensorflow graph
    #
    # sess = tf.Session()
    #
    # gen = gn.create_generator()
    # gen.summary()
    # a = Input(shape=[12,32])
    # gen = gen(a)
    # gen.summary()
    #
    #

    # g_opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # g_loss =
    #
    # disc = ds.create_discriminator()
    # disc.summary()

    # for i in range(epochs):
    #     np.random.shuffle(dataset)
    #     for song in dataset:

def GAN():
    # Start stacking GAN
    sess = tf.Session()
    K.set_session(sess)

    # input for choosing real/fake data
    if_real = tf.placeholder(tf.bool, name="IF_REAL")

    # input for GAN
    inp = tf.placeholder(tf.float32, shape=(1,1,12), name="INPUT")

    # Create generator model
    gen = gn.create_generator(inp)
    # gen.summary()

    # Get output tensor and do some reshaping
    g_out = tf.expand_dims(gen.output, axis=1)

    # If on current step we want to feed discriminator with real data than create
    # tensor from inp. else from result of generator
    x = tf.cond(if_real, lambda: inp, lambda: g_out)

    # Create discriminator model
    dis = ds.create_discriminator(x)
    #dis.summary()

    # Again get output tensor
    d_out = dis.output

    # Calculate discriminator loss
    dloss = - tf.reduce_mean(tf.to_float(if_real) * tf.log(d_out) - (1 - tf.to_float(if_real)) * tf.log(1 - d_out))

    # Calculate generator's loss
    gloss = - tf.reduce_mean(tf.log(1 - d_out))

    # Define optimizer (for learning rate and beta1 see advices in Deep Convolutional GAN pre-print on arXive)
    opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    # Create writer for tensorboard
    summary_writer = tf.summary.FileWriter('/home/kell/tensorflow_logs', graph=sess.graph)

    # Initialize vars
    init = tf.global_variables_initializer()
    sess.run(init)

    # run simple test
    res = sess.run(dloss, feed_dict={if_real: True, inp: np.expand_dims(np.expand_dims(np.array([1,2,3,4,5,6,7,8,9,0,1,2]), axis=0), axis=0)})
    print(res)

    # Close session
    summary_writer.close()
    sess.close()



# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# z = tf.placeholder(tf.float32)
#
# def fn1(a, b):
#   return tf.mul(a, b)
#
# def fn2(a, b):
#   return tf.add(a, b)
#
# pred = tf.placeholder(tf.bool)
# result = tf.cond(pred, lambda: fn1(x, y), lambda: fn2(y, z))
#
# Then you can call it as bellowing:
#
# with tf.Session() as sess:
#   print sess.run(result, feed_dict={x: 1, y: 2, z: 3, pred: True})
#   # The result is 2.0





# # fake data and real data def
# fake = []
# real = []
#
# dloss = 0  #some loss function for discriminator
# gloss = 0  #some loss function for generator
#
# gen = gn.create_generator(12)
# disc = ds.create_discriminator(12)
#
# # Create session for computing
# sess = tf.Session()
#
#
# doptimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(dloss)
# goptimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(gloss)
#
#
# # optimizer could be split like that:
# # optimizer = tf.train.AdamOptimizer()
# # grads_and_vars = optimizer.compute_gradients(loss)
# ## Change grads_and_vars as you wish
# # opt_operation = optimizer.apply_gradients(grads_and_vars)
