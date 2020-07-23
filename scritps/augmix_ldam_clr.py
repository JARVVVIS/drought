## import necessary libs
import os
import numpy as np
import matplotlib.pyplot as plt
from helper.custom_dataset import CustomData
from sklearn.metrics import confusion_matrix
from efficientnet_pytorch import EfficientNet

from LDAM import LDAMLoss
from clr_schedules import triangular_lr
from torch.optim.lr_scheduler import LambdaLR
import torchcontrib
from helper.aug_and_mix import AugMix

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
    
    train_data_aug  = AugMix(train_data)
    
    
    if weighted:
        
        print('Weighted Loader ......')
        
        weights = 1. / torch.tensor(cls_num_list, dtype=torch.float)
        samples_weights = weights[train_targets]
        
        print(samples_weights)

        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)


        train_loader = torch.utils.data.DataLoader(train_data_aug,batch_size=batch_size,sampler=sampler)
        valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size,shuffle=True)
        
        print('Training Images : {}'.format(len(train_loader.sampler)))
        print('Validation Images : {}'.format(len(valid_loader.sampler)))
        
        return train_loader,valid_loader
    
    
    train_loader = torch.utils.data.DataLoader(train_data_aug,batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size,shuffle=True)
        
    print('Training Images : {}'.format(len(train_loader.sampler)))
    print('Validation Images : {}'.format(len(valid_loader.sampler)))
    
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

    base_opt = torch.optim.Adam(model.parameters(),1.) ## base optimizer

    ## Set up the CLR
    step_size = 2
    max_lr = 1e-4
    base_lr= 1e-5
    cyclic_lr = triangular_lr(step_size,base_lr,max_lr)
    scheduler = LambdaLR(base_opt,cyclic_lr)

    ## Setup the SWA 
    opt = torchcontrib.optim.SWA(base_opt)

    ## Training Loop
    n_epochs = 40
    train_loader,valid_loader = load_data(weighted=False)
    valid_acc_max = -np.Inf
    train_losses, valid_losses = [], []
    val_acc = []
    idx = 0
    betas = [0.99,0.99999]


    for epoch in range(1,n_epochs+1):    
            
        train_loss = 0.0 ## running losses
        valid_loss = 0.0

        if epoch>25 and idx==0:
                idx += 1
                valid_acc_max=-np.Inf
                correct,total = acc_by_class(valid_loader,model)

                
        beta = betas[idx]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        
        criterion = LDAMLoss(cls_num_list=cls_num_list,weight=per_cls_weights) ## Loss
        
        
        model.train() ## training mode
        for data,labels in train_loader:
            
            batch_size = data.size(0)
            opt.zero_grad() ## clear the gradient


            images_all,labels = torch.cat(data, 0).to(device),labels.to(device)
            logits_all = model(images_all)
            
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, data[0].size(0))
            
            loss = criterion(logits_clean,labels) ## Loss on original data
            
            p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)
            
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            
            torch.nn.utils.clip_grad_norm_(model.parameters(),1) ## Gradinet clipping
            opt.step()
            train_loss += loss.item()*batch_size
        
        model.eval()
        for data,labels in valid_loader:
            
            data,labels = data.to(device),labels.to(device)
            opt.zero_grad()
            output = model(data)
            loss = criterion(output,labels)
            
            valid_loss += loss.item()*batch_size    
        
    train_loss /= len(train_loader.sampler)
    valid_loss /= len(valid_loader.sampler)
    
    scheduler.step()
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