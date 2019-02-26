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


def trainer(arch, data_dir, save_dir, epochs, learning_rate, machine, hidden_units):
	
	train_dir = data_dir[0] + '/train'
	valid_dir = data_dir[0] + '/valid'
	# test_dir = data_dir + '/test'

	# TODO: Define your transforms for the training, validation, and testing sets
	train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
	                                      transforms.RandomRotation(45),
	                                      transforms.RandomHorizontalFlip(),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406],
	                                                           [0.229, 0.224, 0.225])])
	valid_transforms = transforms.Compose([transforms.Resize(256),
	                                     transforms.CenterCrop(224),
	                                     transforms.ToTensor(),
	                                     transforms.Normalize([0.485, 0.456, 0.406],
	                                                          [0.229, 0.224, 0.225])])
	# TODO: Load the datasets with ImageFolder
	train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
	# test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms)

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
	validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=False)

	with open('cat_to_name.json', 'r') as f:
		cat_to_name = json.load(f)
	idx_to_name = {value:cat_to_name[key] for key, value in train_datasets.class_to_idx.items()}
	

	# TODO: Build and train your network
	if arch == 'vgg16':
		model = models.vgg16(pretrained=True)
	if arch == 'alexnet':
		model = models.alexnet(pretrained=True)

	for param in model.parameters():
	    param.requires_grad = False

	output_size = len(cat_to_name)
	# get the number of input features of the classifier
	features = list(model.classifier.children())[:-1]
	in_features = list()
	for i, feature in enumerate(features):
		if (isinstance(feature, torch.nn.modules.linear.Linear)):
			in_feature = model.classifier[i].in_features
			in_features.append(in_feature)
	classfier = nn.Sequential(OrderedDict([
	                         ('fc1', nn.Linear(in_features[0], hidden_units)),
	                         ('relu1', nn.ReLU()),
	                         ('fc2', nn.Linear(hidden_units, hidden_units)),
	                         ('relu2', nn.ReLU()),
	                         ('fc3', nn.Linear(hidden_units, output_size)),
	                         ('output', nn.LogSoftmax(dim=1))
	                         ]))
	model.classifier = classfier

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

	#Verify availability of GPU
	print("GPU available:", torch.cuda.is_available())

	if machine == 'GPU':
		device = 'cuda'
	else:
		device = 'cpu'

	model.to(device)

	training_loss = 0
	print_every = 30
	step = 0
	epochs = int(epochs)

	for e in range(epochs):
	    model.train()
	    for i, (inputs, labels) in enumerate(trainloaders):
	        step += 1
	        
	        optimizer.zero_grad()
	        inputs, labels = inputs.to(device), labels.to(device)

	        output = model.forward(inputs)
	        loss = criterion(output, labels)
	        loss.backward()
	        optimizer.step()

	        training_loss += loss.item()
	        
	        if (i+1) % print_every == 0:
	            model.eval()
	            
	            with torch.no_grad():
	                valid_loss, accuracy = validation(model, validloaders, criterion, device)
	            print("Epoch {}/{}...Training Loss: {:.3f}...Validation Loss: {:.3f}...Accuracy: {:.3f}"\
	                  .format(e+1, epochs, training_loss/print_every, valid_loss/len(validloaders), accuracy/len(validloaders)))
	            training_loss = 0
	        model.train()

	# TODO: Save the checkpoint 
	checkpoint = {
	              'pretrained_model':model.features,
	              'classifier': model.classifier,
	              'state_dict': model.state_dict(),
	              'optimizer': optimizer.state_dict(),
	              'arch': arch,
	              'hidden_units': hidden_units,
	              'class_to_idx': idx_to_name,
	              'output_units': len(cat_to_name),
	              'epochs': epochs
	             }

	torch.save(checkpoint, 'checkpoint.pth')	
	return model
