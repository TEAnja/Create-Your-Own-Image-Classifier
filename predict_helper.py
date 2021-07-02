import torch
from torchvision import datasets, transforms, models

#from train import *
import json

#LOADING THE EXISTING MODEL, checkpoint
def load_checkpoint(filepath):#, learn_rate):
    #checkpoint = torch.load(filepath)
    checkpoint = torch.load(filepath, map_location=('cpu' if ('gpu' and torch.cuda.is_available()) else 'cpu'))
    
    arch = str(checkpoint['arch'])
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model# ,optimizer

#IMAGE PREPROCESSING
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    img_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]) 
    img_tensor = img_transforms(img)
    return img_tensor


#PREDICT TOP K CLASSES
def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    # TODO: Implement the code to predict the class from an image file
    model = model.to(device)
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    img = img.to(device)
    
    model.eval()
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    with torch.no_grad():   
        output = model.forward(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk)
        top_p, top_class = top_p[0].tolist(), top_class[0].tolist()          
           
        top_classes = [model.idx_to_class[idx] for idx in top_class]
        
        if cat_to_name:
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
            top_classes = [cat_to_name[str(idx)] for idx in top_class]
        
    return top_p, top_classes