# !/usr/bin/python3

# Дискриминатор для GAN
# Принимает на вход вектор (часть такта), пытается угадать,
# пришла ли она от генератора или из реальной композиции.
# Использует (как и генератор) GRU слои для того, чтобы помнить
# предыдущую часть композиции

# Input - 12-bit vector
# Output - 1 bit answer (fake or real data)

from keras.models import *
from keras.layers import Dense, GRU, Activation
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

# expected input data shape: (<batch_size>, <timesteps (how far to look back)>, <data_dim>)
def create_discriminator(inp_tensor):
    with tf.name_scope('discriminator'):
        prev_l = Input(tensor=inp_tensor)
        # prev_l = GRU(12, return_sequences=True, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.5))(prev_l)
        # prev_l = LeakyReLU(alpha=0.2)(prev_l)
        # prev_l = GRU(11, return_sequences=True, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.5))(prev_l)
        # prev_l = LeakyReLU(alpha=0.2)(prev_l)
        prev_l = GRU(12, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.5))(prev_l)
        prev_l = LeakyReLU(alpha=0.2)(prev_l)
        prev_l = Dense(1, kernel_initializer=RandomNormal(mean=0, stddev=0.5))(prev_l)
        prev_l = Activation('tanh')(prev_l)
        return Model(inputs=inp_tensor, outputs=prev_l)