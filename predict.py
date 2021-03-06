import torch
from torchvision import datasets, transforms, models

import argparse

from predict_helper import load_checkpoint, process_image, predict
import json
    
def get_arguments_predict():
    parser = argparse.ArgumentParser('Arguments for the predict.py file')
    parser.add_argument('image_pth', type=str, help='Path to your image.')
    parser.add_argument("checkpoint", type=str, help='Checkpoint of your model.')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely casses.')
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help="Display category names.")
    parser.add_argument('--gpu', action='store_true', help='GPU')

    args = parser.parse_args()   
    return args
    
def main_predict():
    args = get_arguments_predict()
    image = args.image_pth
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name = args.category_names
    checkpoint = args.checkpoint
    
    #model = checkpoint.model
    
    device = torch.device("cuda")
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #LOAD YOUR CHECKPOINT
    #load_checkpoint(checkpoint, lear_rate)
    
    model = load_checkpoint(checkpoint)
    
    #PREPROCESS IMAGE
    image = process(image)
    
    #RETURN TOP K PROBABILITIES & CLASSES
    predict(image, load_checkpoint, top_k, cat_to_name)
    
if __name__ == "__main__":
    main_predict()