# !/usr/bin/python3

# В данном файле описывается генератор
# На вход подается 12-битный вектор нот,
# каждый бит означает какую-то
# ноту (0-й - C, 1-й - C# и тд)
# Один вектор - четверть такта.
# Можно подавать несколько векторов (пол-такта или такт),
# надо подумать.
# Ожидаемый выход - 12-битный вектор (или набор векторов),
# обозначающий следующую четверть такта (или часть такта).
# Задача - наиболее точно предсказывать следующий такт, возможно не
# гарантированно точно, но хотя бы в той же тональности

# Input - 12-bit vector
# Output - 12-bit vector

from keras.models import *
from keras.layers import Dense, LSTM
from keras.initializers import RandomNormal
from keras.constraints import maxnorm
import tensorflow as tf


# expected input data shape: (<batch_size>, <timesteps (how far to look back)>, <data_dim>)
# (1, 1, 12)?
def create_generator(inp_tensor):
    with tf.name_scope('generator'):
        prev_l = Input(tensor=inp_tensor)
        prev_l = LSTM(144, return_sequences=True, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.02), activation='relu')(prev_l)
        prev_l = LSTM(72, return_sequences=True, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.02), activation='relu')(prev_l)
        prev_l = LSTM(36, stateful=True, kernel_initializer=RandomNormal(mean=0, stddev=0.02), activation='relu')(prev_l)
        prev_l = Dense(36, kernel_constraint=maxnorm(4), kernel_initializer=RandomNormal(mean=0, stddev=0.02), activation='relu')(prev_l)
        prev_l = Dense(12, kernel_constraint=maxnorm(4), kernel_initializer=RandomNormal(mean=0, stddev=0.02), activation='tanh')(prev_l)
        return Model(inputs=inp_tensor, outputs=prev_l)
