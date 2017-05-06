import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import generator as gn

import math
import random

import numpy as np

#some data preparation here


#fake data and real data def
fake = []
real = []

gen = gn.create_generator(12)

#
# disc = ds.create_generator(12)