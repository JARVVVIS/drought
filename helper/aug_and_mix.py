'''
Implement Augmix Code with Augmix data wrapper
'''

import numpy as np 
import matplotlib.pyplot as plt 
import augmentation 
from PIL import Image

import torch
import torch.utils.data as data_
import torchvision.transforms as transforms



class AugMix(data_.Dataset):
    '''
    Augmix Dataset Wrapper
    '''
    def __init__(self,dataset,width=3,severity=3):
        self.dataset = dataset 
        self.width   = width
        self.severity = severity
        self.aug_list = augmentation.augmentations
        self.preprocess = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.718,0.734,0.675),(0.172,0.168,0.195))
    ])

    def aug(self,image):
        ws = np.float32(np.random.dirichlet([1]*self.width))
        m  = np.float32(np.random.beta(1,1))
        mix = torch.zeros_like(self.preprocess(image))

        for i in range(self.width):
            ## For each chain
            image_aug = image.copy()
            depth = np.random.randint(1,4) ## Depth for a particluar chain
            for _ in range(depth):
                op = np.random.choice(self.aug_list) ## Choose an augmentation operation
                image_aug = op(image_aug,self.severity) ## Do augmentation -> stack augmentations on top of each other for a particular chain
            mix += ws[i]*self.preprocess(image_aug)
        
        mixed = (1-m)*self.preprocess(image) + m*mix
        return mixed
    
    def __len__(self):
        '''
        Length of Dataset
        '''
        return 86317
    


    def __getitem__(self,index):
        x,y = self.dataset[index]
        im_tuple = (self.preprocess(x),self.aug(x),self.aug(x))
        return (im_tuple,y) ## originial+augmented_1+augmented_2 with label as y


def implot(image):
    plt.figure(figsize=(8,8))
    image = image.numpy()
    image = np.transpose(image,(1,2,0))
    var = [0.172,0.168,0.195]
    mean = [0.718,0.734,0.675]
    image= image*var + mean
    im = Image.fromarray((image*255).astype(np.uint8))
    Image._show(im)

if __name__ == '__main__':
    pass


