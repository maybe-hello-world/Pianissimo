# !/usr/bin/python3

# Input - 12-bit vector
# Output - 12-bit vector

from keras.models import *
from keras.layers import Dense, GRU, Activation, Dropout
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf


# expected input data shape: (<batch_size>, <timesteps (how far to look back)>, <data_dim>)
def create_generator(inp_tensor):
    with tf.name_scope('generator'):
        prev_l = Input(tensor=inp_tensor)
        prev_l = GRU(18, return_sequences=True, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.02))(prev_l)
        prev_l = Dropout(rate=0.45)(prev_l)
        prev_l = LeakyReLU(alpha=0.2)(prev_l)
        prev_l = GRU(16, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.3))(prev_l)
        prev_l = Dropout(rate=0.45)(prev_l)
        prev_l = LeakyReLU(alpha=0.2)(prev_l)
        prev_l = Dense(12, kernel_initializer=RandomNormal(mean=0, stddev=0.2))(prev_l)
        prev_l = Activation('tanh')(prev_l)
        return Model(inputs=inp_tensor, outputs=prev_l)
