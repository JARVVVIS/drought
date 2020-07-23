import torch
import os
import torch.utils.data as data_
import numpy as np
import glob


class CustomData(data_.Dataset):
    
    def __init__(self,root,mode='train',transform=None,bands=None):
        '''
        Custom DataSet class for DroughtWatch
        '''
        self.mode = mode
        self.root = root
        self.train_fol = 'train'
        self.val_fol   = 'val'
        self.classes = ['0','1','2','3']
        self.transform = transform
        self.bands = bands
        
        self.train_samples = glob.glob(root+'/train/*/*') ## list of all training samples
        self.val_samples   = glob.glob(root+'/val/*/*')   ## list of all validation samples
        
        self.targets = []
        for file in self.train_samples:
            label = int(file.split('/')[2])
            self.targets.append(label)

    def load_bands(self,path):
        '''
        Load Particular Bands from Multispectral Image
        None -> all bands
        '''
        image = np.load(path) ## Load the Image for this particular path
        
        if self.bands:
            image = image[:,:,self.bands] ## Load in specific bands
            
        if self.transform:
            image = self.transform(image)
            return image
        
        image  = torch.from_numpy(image).float()
        image = np.transpose(image,(2,0,1)) ## W*H*C -------> C*W*H
        return image
    
    
    
    def __len__(self):
        '''
        Length of Dataset
        '''
        if self.mode == 'train':
            count = len(self.train_samples)
        else:
            count = len(self.val_samples)
        return count
    
    
    
    def __getitem__(self,index):
        '''
        Load in Image-Label Pair
        '''
        if self.mode == 'train':
            files = self.train_samples
        elif self.mode == 'val':
            files = self.val_samples

        path = files[index]
        image = self.load_bands(path)
        label = int(path.split('/')[2])
        return (image,label)        