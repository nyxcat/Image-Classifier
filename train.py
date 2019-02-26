import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image
import json
from model_train import trainer
from helper import *

import argparse
# CALL something like this:  python train.py /flowers/ --machine=ccc

parser = argparse.ArgumentParser(description='Image Classifier Application Trainer.')
parser.add_argument('data_dir', action='store', nargs='*', default='flowers')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='/checkpoint.pth')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16')
parser.add_argument('--learning_rate', action='store', dest='learning_rate', default=0.001)
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=1024)
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
parser.add_argument('--machine', action='store', dest='machine', default='GPU')

args = parser.parse_args()


def main():

	trainer(args.arch, args.data_dir, args.save_dir, args.epochs, args.learning_rate, args.machine, args.hidden_units)

if __name__ == '__main__':
	main()