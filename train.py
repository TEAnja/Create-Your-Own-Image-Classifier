import argparse
from train_helper import get_data, new_model, train_model, test_model, save_checkp

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def get_arguments():
    parser = argparse.ArgumentParser('Arguments for the test.py file')
    parser.add_argument('data_dir', type=str, help='Your data directory.')
    parser.add_argument('--arch', type=str, default='densenet121', help='What model will you be using?')
    parser.add_argument('--save_dir', action='store_true', help='Save your checkpoints')
    parser.add_argument("--save_to", type=str, default="ImageClassifier/checkpoint.pth", help="Checkpoint path.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model.')
    parser.add_argument('--hidden_units', type=int, default=700, help='Number of hidden units.')
    parser.add_argument('--hidden_units2', type=int, default=400, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--gpu', action='store_true', help='GPU')

    args = parser.parse_args()   
    return args

arch = get_arguments().arch

def main_train():
    #GET ARGUMENTS FROM THE USER
    args = get_arguments()
    
    arch = args.arch
    model = new_model(arch)
    learn_rate = args.learning_rate
    epochs = args.epochs
    data_dir = args.data_dir
    hidden_units = args.hidden_units
    hidden_units2 = args.hidden_units2
    save_to = args.save_to
    
    device = torch.device('cuda')
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = get_data(data_dir)
    train_datasets = data[0]
    valid_datasets = data[1]
    test_datasets = data[2]
    
    train_loader = data[3]
    valid_loader = data[4]
    test_loader = data[5]
    
    #TRAIN THE MODEL
    model = train_model(model, learn_rate, epochs, train_loader, valid_loader, device, hidden_units, hidden_units2)
    print("knockknock")
    #TEST THE MODEL
    tested_model = test_model(model, test_loader, args.gpu)
    print("whos there?!")
    #SAVE THE CHECKPOINT    
    if args.save_dir:
        save_checkp(model, train_datasets, epochs, save_to, arch, learn_rate)

    
if __name__ == "__main__":
    main_train()