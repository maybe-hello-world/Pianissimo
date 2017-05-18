# !/usr/bin/python3

import sys
import getopt
import os
import datetime

import trainer
import tester

## Some constants
g_weights_name = 'g_weights'
d_weights_name = 'd_weights'

try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:i:")
except getopt.GetoptError:
    print('main.py -o <train|test> -i <inputfolder>')
    sys.exit(2)

inputfolder = ""
train = True
for opt, arg in opts:
    if opt == '-h':
        print('main.py -i <inputfolder>')
        sys.exit(0)
    elif opt == '-o':
        if arg in ('test', 'train'):
            train = arg == 'train'
        else:
            print("Unknown operation")
            sys.exit(1)
    elif opt in ("-i", "--ifolder"):
        inputfolder = arg

if len(inputfolder) == 0 or not os.path.isdir(inputfolder):
    print("Folder is missing")
    sys.exit(1)

print(datetime.datetime.now())
if train:
    trainer.train(inputfolder, 4)
else:
    tester.test()
print(datetime.datetime.now())