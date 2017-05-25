# !/usr/bin/python3

import sys
import argparse
import os

from config import *

from trainer import train
from tester import test

parser = argparse.ArgumentParser(description='Pianissimo project')

subparsers = parser.add_subparsers(dest='command', title="Commands")
train_parser = subparsers.add_parser('train', help='Start training of existing (or new created) model')
train_parser.add_argument('-i', type=str, required=True, help="Folder with parsed music data", metavar='INPUTFOLDER')

test_parser = subparsers.add_parser('test', help='Start creating sequence from start_sequence from config.py')

args = parser.parse_args(sys.argv[1:])
inputfolder = ""

if args.command == "train":
    inputfolder = args.i
    if len(inputfolder) == 0 or not os.path.isdir(inputfolder):
        print("Folder is missing")
        sys.exit(1)

if args.command == "train":
    train(inputfolder)
elif args.command == "test":
    test()
else:
    sys.exit(1)