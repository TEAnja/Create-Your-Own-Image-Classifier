#import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np

def get_data(data_dir):
    train_dir = 'ImageClassifier/' + data_dir + '/train'
    valid_dir = 'ImageClassifier/' + data_dir + '/valid'
    test_dir = 'ImageClassifier/' + data_dir + '/test'

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    to_return = [train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader]
    return to_return

def new_model(arch):
   model = getattr(models, arch)(pretrained=True)
   return model

#CONSTRUCTING A CLASSIFIER
def new_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(1024, 600),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.3),
                                         nn.Linear(600, 400),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.3),
                                         nn.Linear(400, 102),
                                         nn.LogSoftmax(dim=1))
        return(model.classifier)
        
#SETTING AN OPTIMIZER
def new_optimizer(classifier, learn_rate):
    print('new_optimizer at your service')
    return (optim.Adam(classifier.parameters(), lr=learn_rate))
    
#TRAINING THE MODEL
def train_model(model, learn_rate, epochs, train_loader, valid_loader, device):
    print('Welcome to the train_model!')
    classifier = new_classifier(model)
    criterion = nn.NLLLoss()
    optimizer = new_optimizer(classifier, learn_rate)
    model.to(device);      
    
    steps = 0
    running_loss = 0
    print_every = 64
    for epoch in range(epochs):
        print('You are in the training loop')
        for inputs, labels in train_loader:
            inputs = inputs.float()
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            print('a')

            optimizer.zero_grad()
            print('e')

            output = model.forward(inputs)
            print('1')
            loss = criterion(output, labels)
            print('2')
            loss.backward()
            print('3')
            optimizer.step()
            print('o')
            running_loss += loss.item()
            print('u')

            if steps % print_every == 0:
                print('You are in the printing loop')
                validate_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs = inputs.float()
                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        validate_loss += batch_loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}",
                      f"Training loss: {running_loss/print_every:.3f}",
                      f"Validation loss: {validate_loss/len(valid_loader):.3f}",
                      f"Validation Accuracy: {accuracy/len(valid_loader):.3f}")

                running_loss = 0
                model.train()


#TESTING THE MODEL
def test_model(model, testloader):
    print("I'm here!")
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            model.eval()
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(#f"Testing loss: {test_loss/len(testloader):.3f}",
              f"Test Accuracy: {accuracy/len(testloader):.3f}")
    
    
    
#SAVING THE CHECKPOINT
def saving_checkp(model, train_datasets, epochs):
    checkpoint= {'classifier' : model.classifier,
                 'class_to_idx' : train_datasets.class_to_idx,
                 'state_dict' : model.state_dict(),
                 'optimizer_state_dict' : optimizer.state_dict(),
                 'epochs' : epochs,
                 }

    torch.save(checkpoint, 'checkpoint.pth')

