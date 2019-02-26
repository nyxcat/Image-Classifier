
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image

def validation(model, validloaders, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in validloaders:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        newmodel = models.vgg16(pretrained=True)
    if checkpoint['arch'] == 'alexnet':
        newmodel = models.alexnet(pretrained=True)
    for param in newmodel.parameters():
        param.requires_grad = False
    hidden_units = checkpoint['hidden_units']
    output_units = checkpoint['output_units']
    # get the number of input features of the classifier
    features = list(newmodel.classifier.children())[:-1]
    in_features = list()
    for i, feature in enumerate(features):
        if (isinstance(feature, torch.nn.modules.linear.Linear)):
            in_feature = newmodel.classifier[i].in_features
            in_features.append(in_feature)
        
    newmodel.classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(in_features[0], hidden_units)),
                         ('relu1', nn.ReLU()),
                         ('fc2', nn.Linear(hidden_units, hidden_units)),
                         ('relu2', nn.ReLU()),
                         ('fc3', nn.Linear(hidden_units, output_units)),
                         ('output', nn.LogSoftmax(dim=1))
                         ]))

    newmodel.load_state_dict(checkpoint['state_dict'])
    newmodel.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epochs']
    print("Loaded checkpoint '{}' (epoch {})"
                  .format(filepath, checkpoint['epochs']))
    return newmodel


# TODO: Process a PIL image for use in a PyTorch model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    ratio = width / height
    if ratio < 1:
        new_width = 256
        new_height = int(new_width / ratio)
    else:
        new_height = 256
        new_width = int(new_height * ratio)
        
    im = im.resize((new_width, new_height))
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = (new_width + 224)/2
    bottom = (new_height + 224)/2
    im = im.crop((left, top, right, bottom))

    np_image = np.array(im)
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image / np.max(np_image)

    np_image = np_image.transpose()
    
    return np_image
    

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)
    c_to_idx = model.class_to_idx
    model.eval()
    with torch.no_grad():
        output = model.forward(image.to(device))
    ps = torch.exp(output)
    tops = ps.topk(topk)
    ps = tops[0].cpu().numpy()
    indexes = tops[1].cpu().numpy()

    probs = list()
    names = list()
    for p,i in zip(ps[0], indexes[0]):
        probs.append(p)
        names.append(c_to_idx[i])
    return probs, names



