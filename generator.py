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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.constraints import maxnorm


def create_generator(input_dim):
    model = Sequential()
    model.add(LSTM(144, init='uniform', input_dim=(input_dim,), activation='tanh', W_constraint=maxnorm(4)))
    return model
