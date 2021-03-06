## import necessary libs
import os
import numpy as np
import matplotlib.pyplot as plt
from helper.custom_dataset import CustomData
from sklearn.metrics import confusion_matrix
from efficientnet_pytorch import EfficientNet

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data.sampler import WeightedRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

## Update the class_number_list
cls_num_list = [0,0,0,0]
for i in range(4): 
    path, dirs, files = next(os.walk("./dataset/train/"+str(i)+'/'))
    cls_num_list[i] = len(files)

def load_data(root = 'dataset',num_bands=3,batch_size=32,weighted=True):

    if num_bands==3:
        train_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.718,0.734,0.675),(0.172,0.168,0.195))
                            ])
    
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.718,0.734,0.675),(0.172,0.168,0.195))
                            ])
    else:

        train_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.720, 0.675, 0.644, 0.654, 0.734, 0.718, 0.679, 0.640, 0.249, 0.969, 0.971),(0.197, 0.195, 0.186, 0.192, 0.168, 0.172, 0.183, 0.187, 0.417, 0.108, 0.108))
                                ])
    
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.720, 0.675, 0.644, 0.654, 0.734, 0.718, 0.679, 0.640, 0.249, 0.969, 0.971),(0.197, 0.195, 0.186, 0.192, 0.168, 0.172, 0.183, 0.187, 0.417, 0.108, 0.108))
        
                                ])


    train_data = CustomData(root,mode='train',transform=train_transform)
    valid_data = CustomData(root,mode='val',transform=valid_transform)

    train_targets = train_data.targets
    
    
    if weighted:
        
        weights = 1. / torch.tensor(cls_num_list, dtype=torch.float)
        samples_weights = weights[train_targets]
        
        print(samples_weights)

        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)


        train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,sampler=sampler)
        valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size,shuffle=True)
        
        return train_loader,valid_loader
    
    
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size,shuffle=True)
    
    
    return train_loader,valid_loader

def accuracy(loader,model):
    correct = 0
    model.eval()
    for data,labels in loader:
        data,labels = data.to(device),labels.to(device)
        output = model(data)
        _,pred = torch.max(output,1) ## get the predictions
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct  += sum(np.squeeze(correct_tensor.cpu().numpy()))
    return correct/len(loader.sampler)

def acc_by_class(loader,model):
    ## calculate per-class accuracy
    correct = [0,0,0,0]
    total = [0,0,0,0]
    model.eval()
    for data,labels in loader:
        data,labels = data.to(device),labels.to(device)
        output = model(data)
        _,pred = torch.max(output,1) ## get the predictions
        correct_tensor = pred.eq(labels.data.view_as(pred))
        
        correct_np = correct_tensor.cpu().numpy().squeeze()
        labels = labels.cpu().numpy().squeeze()
        
        for pred,label in zip(correct_np,labels):
            correct[label] += pred
            total[label] += 1 ## count total no of examples
    
    return correct,total

def target_list(loader,model):
    ## calculate per-class accuracy
    pred_list = []
    true_list = []
    model.eval()
    for data,labels in loader:
        data,labels = data.to(device),labels.to(device)
        output = model(data)
        _,pred = torch.max(output,1) ## get the predictions
        pred_np = pred.detach().cpu().numpy().squeeze()
        labels = labels.cpu().numpy().squeeze()
        
        pred_list.extend(pred_np)
        true_list.extend(labels)
        
    return np.asarray(pred_list),np.asarray(true_list)


def main():
    model_choice = input()
    num_bands    = int(input())

    if model_choice == 'Resnet-18':
        model = models.resnet18(pretrained=True)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat,4)

        model.conv1 = nn.Sequential(nn.ConvTranspose2d(num_bands,32,2,1,2,bias=False),
                                    nn.Conv2d(32,3,1,bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3,64,7,2,3,bias=False))

        model.to(device)

    elif model_choice == 'Resnet-50':
        model = models.resnet50(pretrained=True)
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat,4)

        model.conv1 = nn.Sequential(nn.ConvTranspose2d(num_bands,32,2,1,2,bias=False),
                                    nn.Conv2d(32,3,1,bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(3,64,7,2,3,bias=False))
        model.to(device)
    
    elif model_choice == 'EffcientNet-B4':
        model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=4)
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5) ## the optimizer
    criterion = nn.CrossEntropyLoss()
    ## Training Loop
    n_epochs = 40
    train_loader,valid_loader = load_data(weighted=False)
    valid_acc_max = -np.Inf
    train_losses, valid_losses = [], []
    val_acc = []

    for epoch in range(1,n_epochs+1):    
            
        train_loss = 0.0 ## running losses
        valid_loss = 0.0
        
        model.train() ## training mode
        for data,labels in train_loader:
            
            data,labels = data.to(device),labels.to(device)
            batch_size = data.size(0)
            
            optimizer.zero_grad() ## clear the gradient
            output = model(data) ## get the output
            loss = criterion(output,labels) ## get the output
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),1) ## Gradinet clipping
            optimizer.step()
            train_loss += loss.item()*batch_size
        
        model.eval()
        for data,labels in valid_loader:
            
            data,labels = data.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,labels)
            
            valid_loss += loss.item()*batch_size    
        
    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)
    
    
    valid_acc = accuracy(valid_loader,model)
    
    valid_losses.append(valid_loss)
    train_losses.append(train_loss)
    
    val_acc.append(valid_acc)
    
    print('Epoch : {} \tTraining Loss : {} \tValidation Loss :{} \tValidation Acc : {}'.format(epoch,train_loss,valid_loss,valid_acc))
    print('-'*100)
    if valid_acc >= valid_acc_max:
        print('Validation Acc. increased ({:.6f} --> {:.6f})'.format(
        valid_acc_max,
        valid_acc))
        valid_acc_max = valid_acc


if __name__ == '__main__':
    main()