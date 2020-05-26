import torchvision
from torchvision import datasets, transforms, models
import argparse
import json
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
from PIL import Image

def preproces(data_dir):

    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),

                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'trainloader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   'validloader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)}
    
    return dataloaders, image_datasets

def create_model(arch, hidden_units, lr, gpu):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_units = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr= lr)

 
    return model, criterion, optimizer

def train_model(model, dataloaders, criterion, optimizer, epochs, gpu):
    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, test_losses = [], []
    for epoch in range(epochs):
       for inputs, labels in dataloaders['trainloader']:
        steps += 1
        device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
        model.to(device)  
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['validloader']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(dataloaders['validloader']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['validloader']):.3f}")
            running_loss = 0
            model.train()
    return model

def save_model(save_dir, image_datasets, arch, model, epochs, lr, hidden_units, gpu):
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    model.cpu
    torch.save({'arch': arch,
                'hidden_units': hidden_units,
                'epochs': epochs,
                'lr': lr,
                'gpu': gpu,
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx}, 
                save_dir)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model,_,_ = create_model(checkpoint['arch'],
                             checkpoint['hidden_units'],
                             checkpoint['lr'],
                             checkpoint['gpu'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
   
    return model

def preprocess_image(image):
    
    im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 

    im_transformed = transform(im)
    array_im = np.array(im_transformed)
    np_image = torch.from_numpy(array_im).type(torch.FloatTensor)
    
    return np_image

def predict(image_path, model, gpu, topk=5):
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
       
    img = preprocess_image(image_path)
    img = img.to(device)
    img = img.unsqueeze(0)  
    if gpu == True:
        with torch.no_grad():
            model = model.cuda()
            logits = model.forward(img.cuda())
            probs, probs_labels = torch.topk(logits, topk)
            probs = probs.exp() 
            class_to_idx = model.class_to_idx
    else:
       with torch.no_grad():
            logits = model.forward(img) 
            probs, probs_labels = torch.topk(logits, topk)
            probs = probs.exp() 
            class_to_idx = model.class_to_idx
   
    probs = probs.cpu().numpy()
    probs_labels = probs_labels.cpu().numpy()
    
    classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
    
    classes_list = list()
    
    for label in probs_labels[0]:
        classes_list.append(cat_to_name[classes_indexed[label]])
        
    return (probs[0], classes_list)


def imgshow(image, ax=None, title=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def load_mapping(filepath):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name=json.load(f)

    return cat_to_name

def plot_solution(image_path, model, gpu):
    
    probs, classes = predict(image_path, model, gpu)
    
    image = preprocess_image(image_path)
    axs = imgshow(image, ax = plt)
    index=image_path.split('/')[2]
    plt.title(cat_to_name[str(index)])

    
    plt.figure(figsize=(4,4))
    y_pos = np.arange(len(classes))
    
    performance = np.array(probs)
    
    plt.barh(y_pos, performance, align='center',
        color=sns.color_palette()[0])
    
    y_pos, classes
    plt.yticks(y_pos, classes)
    plt.gca().invert_yaxis()
    print(probs)
