#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR


import argparse
import logging
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# logging configs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))





def test(model, test_loader, criterion, device, epoch_no):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #model.to('cpu')
    logger.info(f"Epoch: {epoch_no} - Testing Model on Complete Testing Dataset!")
    model.eval()
    running_loss = 0
    running_corrects = 0
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) 
            running_corrects += pred.eq(labels.view_as(pred)).sum().item()

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc))

def train(model, train_loader, criterion, optimizer, device, epoch_no):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    #model = model.to(device)
    logger.info(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset!") 
    since = time.time()
    model.train()
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        running_loss += loss.item() * inputs.size(0) #calculate the running loss
        running_corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate the running corrects
        running_samples += len(inputs) #keep count of running samples
        loss.backward()
        optimizer.step()
        if running_samples %500 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*(running_corrects/ running_samples)
            ))
    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects/ len(train_loader.dataset)
    time_elapsed = time.time() - since
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), in {:.0f}m {:.0f}s\n".format(
        total_loss, running_corrects, len(train_loader.dataset), 100.0 * total_acc,  time_elapsed // 60, time_elapsed % 60
    ))  
    
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False 
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256),
                             nn.ReLU(inplace = True),
                             nn.Dropout(0.25),
                             nn.Linear(256, 133),
                             nn.ReLU(inplace = True))
    
    
    return model

def create_data_loaders(data , batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get train and test data loaders")
    train_dataset_path = os.path.join(data, "train")
    test_dataset_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    

        
        
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle= False)
    
    return train_data_loader, test_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    logger.info(f"Running on Device {device}")
    logger.info(f"Hyperparameters : LR: {args.lr}, Eps: {args.eps}, Weight-decay: {args.weight_decay}, Batch Size: {args.batch_size}, Epoch: {args.epochs}")
    logger.info(f"Data Dir Path: {args.data_dir}")
    logger.info(f"Model Dir  Path: {args.model_dir}")
    logger.info(f"Output Dir  Path: {args.output_dir}")
    
        
    
    model=net()
    model = model.to(device)
    
    
    train_data_loader, test_data_loader = create_data_loaders(args.data_dir, args.batch_size )
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)

        # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma= 0.7)
    for epoch_no in range(1, args.epochs +1 ):
        '''
        TODO: Call the train function to start training your model
        Remember that you will need to set up a way to get training data from S3
        '''
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model=train(model, train_data_loader, loss_criterion, optimizer, device, epoch_no)
        '''
        TODO: Test the model to see its accuracy
        '''
        logger.info(f"Epoch {epoch_no} - Starting Testing phase.")
        test(model, test_data_loader, loss_criterion, device, epoch_no)
        scheduler.step()
    
    logger.info("Starting to Save the Model")
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Completed Saving the Model")
    


if __name__=='__main__':

    '''
    TODO: Specify any training args that you might need
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
    parser.add_argument( "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)" )
    parser.add_argument( "--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)" )
                        
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()

    main(args)
