import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse

# from PIL import Image
# import numpy as np

from train import *
from predict_helper import *
import json
    
    
def get_arguments_predict():
    parser = argparse.ArgumentParser('Arguments for the predict.py file')
    parser.add_argument('image_dir', type=str, help='Path to your image.')
    parser.add_argument("checkpoint", type=str, help='Checkpoint of your model.')
    parser.add_argument('--top_k', type=int, default=5; help='Top K most likely casses.')
    parser.add_argument('--category_names', type=int, default="cat_to_name.json", help="Display category names.")
    parser.add_argument('--gpu', action='store_true', help='GPU')

    args = parser.parse_args()   
    return args
    
def main_predict():
    args = get_arguments_predict():
    image = args.image_dir
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name = args.category_names
    
    model = main.model
    
    device = torch.device("cuda")
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #LOAD YOUR CHECKPOINT
    load_checkpoint(image, checkpoint, model)
    
    #PREPROCESS IMAGE
    image = process(image)
    
    #RETURN TOP K PROBABILITIES & CLASSES
    predict(image, load_checkpoint, top_k, cat_to_name)
    
if __name__ == "__main__":
    main_predict()