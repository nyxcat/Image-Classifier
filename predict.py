import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image
import json
from helper import *
import argparse


parser = argparse.ArgumentParser(description='Image Classifier Application Predictor.')
parser.add_argument('image_path', action='store', nargs='*', default='flowers/test/1/image_06752.jpg')
parser.add_argument('checkpoint', action='store', nargs='*', default='checkpoint.pth')
parser.add_argument('--top_k', action='store', dest='k', type=int, default=5)
parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
parser.add_argument('--machine', action="store", dest="machine", default="GPU")

args = parser.parse_args()


def main():

	model = load_checkpoint(args.image_path[1])
	if args.machine == 'GPU':
		device = 'cuda'
	else:
		device = 'cpu'
	model.to(device)
	#imshow(args.image_path[0], ax=None, title=None)

	
	x = np.arange(args.k)
	filepath = 'flowers/test/1/image_06743.jpg'
	probs, classes = predict(args.image_path[0], model, device, args.k)
    

	print(probs, classes)

if __name__ == '__main__':
	main()