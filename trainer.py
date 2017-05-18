# !/usr/bin/python3
from config import *

import tensorflow as tf
import keras.backend as K

import generator as gn
import discriminator as ds

import random
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

from keras.utils import plot_model

def train(inputfolder, epochs):
    ## -----------------------------------------------------------
    ## Dataset preparation
    #

    files = [join(inputfolder, f) for f in listdir(inputfolder) if isfile(join(inputfolder, f))]

    # Func for turning strings like '0, 3, 11' into [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    def to_slice(string):
        ans = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for j in [int(s) for s in string.split(',')]:
            ans[j] = 1
        return np.array(ans)

    dataset = []
    # Read strings, turn them into np.arrays and add dims (output - (1, 1, 12))
    for file in files:
        f = open(file, 'r')
        dataset.append(np.array(
            [
                np.expand_dims(np.expand_dims(to_slice(st.rstrip()), axis=0), axis=0)
                for st in f.readlines()
            ]
        ))
        f.close()

    dataset = np.array(dataset)
    np.random.shuffle(dataset)

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
    if_real = tf.placeholder(tf.bool, name="IF_REAL_INPUT")

    # input for GAN
    inp = tf.placeholder(tf.float32, shape=(1, 1, 12), name="DATA_INPUT")

    # Create generator model
    gen = gn.create_generator(inp)
    if isfile(config['base_folder'] + "/" + config['gen_weights']):
        gen.load_weights(config['base_folder'] + "/" + config['gen_weights'])


    with tf.name_scope("condition"):
        # Get output tensor and do some reshaping
        g_out = gen.output

        # from tanh (-1, 1) to {0, 1}
        g_out = tf.sign(g_out)
        g_out = tf.add(g_out, tf.constant(1.0))
        g_out = tf.div(g_out, tf.constant(2.0))

        # Get output tensor and do some reshaping
        g_out = tf.expand_dims(g_out, axis=1)

        # If on current step we want to feed discriminator with real data then create
        # tensor from inp. else from result of generator
        x = tf.cond(if_real, lambda: inp, lambda: g_out)

    # Create discriminator model
    dis = ds.create_discriminator(x)
    if isfile(config['base_folder'] + "/" + config['dis_weights']):
        dis.load_weights(config['base_folder'] + "/" + config['dis_weights'])

    # Again get output tensor
    d_out = dis.output

    # Finish stacking GAN: add loss functions and different updates
    # So my_gan is a function with such signature:
    # my_gan(if_real=<if_real value>, inp=<12-bit vector value>,
    # d_out=<output of discriminator>, dis=<discriminator model>, gen=<generator model>)
    my_gan = GAN(if_real=if_real, inp=inp, d_out=d_out, dis=dis, gen=gen)

    #
    ## -----------------------------------------------------------

    # Create writer for tensorboard
    summary_writer = tf.summary.FileWriter(config['tensorflow_logs'], graph=sess.graph)

    # Initialize vars
    init = tf.global_variables_initializer()
    sess.run(init)

    # Main cycle
    gloss_history = []
    dloss_history = []
    for i in range(epochs):
        accum_g = 0
        accum_d = 0
        print("Epoch: ", i)
        print("Gen loss, Dis loss")
        cnt = 0
        for song in dataset:
            # x - one song, numpy.array of strings
            #print(cnt)
            losses = [0, 0]
            for slc in song:
                step_bool = random.random() > 0.5
                step = my_gan(sess=sess, vector_input=slc, if_real_input=step_bool)
                losses[0] += step[1][0]  # gen loss
                losses[1] += step[1][1]  # dis loss
                #debug
                if np.isnan(losses[0]) or np.isnan(losses[1]):
                    print(step[2])  #output of discriminator

            accum_g += losses[0]/len(song)
            accum_d += losses[1]/len(song)
            print("\n" + str(cnt))
            print(losses[0]/len(song), losses[1]/len(song))

            # After feeding entire song reset models
            gen.reset_states()
            dis.reset_states()
            cnt += 1

        gloss_history.append(accum_g)
        dloss_history.append(accum_d)
        print("Gen loss: %d, Dis loss: %d" % (accum_g, accum_d))

    # Save generator description...
    with open(config['base_folder'] + "/" + config['gen_model'], "w") as file:
        file.write(gen.to_yaml())
    # ...PNG visualization...
    plot_model(gen, to_file=config['base_folder'] + "/" + config['gen_picture'], show_shapes=True)
    # ...and weights
    gen.save_weights(config['base_folder'] + "/" + config['gen_weights'])

    # Similar with discriminator
    with open(config['base_folder'] + "/" + config['dis_model'], "w") as file:
        file.write(dis.to_yaml())
    plot_model(dis, to_file=config['base_folder'] + "/" + config['dis_picture'], show_shapes=True)
    dis.save_weights(config['base_folder'] + "/" + config['dis_weights'])

    # Close session
    summary_writer.close()
    sess.close()

    plt.plot(range(0, epochs), gloss_history, color='red', linewidth=2.)
    plt.plot(range(0, epochs), dloss_history, color='blue', linewidth=2.)
    plt.show()

#g_out is for debug
def GAN(if_real, inp, d_out, dis, gen):
    with tf.name_scope("output"):
        # d_out uses tanh activation, so it's in (-1, 1) but we need (0, 1)
        nd_out = tf.div(tf.add(d_out, tf.constant(1.0)), tf.constant(2.0))

        # Without 0's and 1's 'cause we don't like NaNs in loss function
        nd_out = tf.multiply(nd_out, tf.constant(0.99)) + tf.constant(0.005)

        # Calculate discriminator loss
        dloss = tf.reduce_mean(tf.to_float(if_real) * tf.log(nd_out) + (1 - tf.to_float(if_real)) * tf.log(1 - nd_out))

        # Calculate generator's loss
        gloss = tf.reduce_mean(tf.log(1 - nd_out))

    dloss = tf.negative(dloss, name="DLOSS")
    gloss = tf.negative(gloss, name="GLOSS")

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

    # We have to update all other tensors like batch_normalization or stateful LSTM nodes transition
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
        nonlocal train_step, losses, if_real, inp, dis, d_out

        res = sess.run([train_step, losses, d_out], feed_dict={
            if_real: if_real_input,
            inp: vector_input,
        })

        return res

    return gan_feed