import argparse
from train import *
#from predict import *

def get_arguments():
    parser = argparse.ArgumentParser('Arguments for the test.py file')
    parser.add_argument('data_dir', type=str, help='Your data directory.')
    parser.add_argument('--arch', type=str, default='densenet121', help='What model will you be using?')
    parser.add_argument('--save_dir', action='store_true', help='Save your checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--gpu', action='store_true', help='GPU')

    args = parser.parse_args()
    data_dir = args.data_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    
    return args

def main():
    #GET ARGUMENTS FROM THE USER
    args = get_arguments()
    
    model = new_model(args.arch)
    learn_rate = args.learning_rate
    epochs = args.epochs
    data_dir = args.data_dir
    
    device = torch.device('cpu')
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
    trained_model = train_model(model, learn_rate, epochs, train_loader, valid_loader, device)
    
    #TEST THE MODEL
    tested_model = test_model(trained_model, test_loader)
    
    #SAVE THE CHECKPOINT
    save_checkpoint = save_checkp(tested_model, train_datasets, epochs)
    
    if args.save_dir:
        saving_checkp(model, train_datasets, epochs)

    
if __name__ == "__main__":
    main()