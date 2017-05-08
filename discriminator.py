# !/usr/bin/python3

# Дискриминатор для GAN
# Принимает на вход вектор (часть такта), пытается угадать,
# пришла ли она от генератора или из реальной композиции.
# Используте (как и генератор) LSTM (GRU?) слои для того, чтобы помнить
# предыдущую часть композиции

# Input - 12-bit vector
# Output - 1 bit answer (fake or real data)

from keras.models import *
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.constraints import maxnorm

# expected input data shape: (<bacth_size>, <timesteps (how far to look back)>, <data_dim>)
def create_discriminator(inp_tensor):
    prev_l = Input(tensor=inp_tensor)
    prev_l = LSTM(144, return_sequences=True, stateful=True, kernel_initializer='uniform')(prev_l)
    prev_l = LSTM(72, return_sequences=True, stateful=True, kernel_initializer='uniform')(prev_l)
    prev_l = LSTM(36, stateful=True, kernel_initializer='uniform')(prev_l)
    prev_l = Dense(1, kernel_constraint=maxnorm(4), kernel_initializer='uniform')(prev_l)
    prev_l = Activation('tanh')(prev_l)
    return Model(inputs=inp_tensor, outputs=prev_l)
