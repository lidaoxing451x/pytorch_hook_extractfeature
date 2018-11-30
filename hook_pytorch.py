from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
'''#loaddata
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),'val': transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
data_dir = 'hymenoptera_data1'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
'''

data_transform=transforms.Compose([transforms.RandomCrop(224, padding=4) ,transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image_dataset=datasets.ImageFolder('hymenoptera_data1/test/',data_transform)
dataloader=torch.utils.data.DataLoader(image_dataset,batch_size=4,shuffle=False, num_workers=0)
resnet18 = torchvision.models.resnet18(pretrained=True)#download resnet18
resnet18.eval()
'''
class LayerActivations():
#hook:use to Visualizing intermediate CNN layers
    features=None
    def __init__(self,model,layer_name):
        self.hook = model[layer_name].register_forward_hook(self.hook_fn)
    
    def hook_fn(self,module,input,output):
        self.features = output.cpu().data.numpy()
    
    def remove(self):
        self.hook.remove()
conv_out = LayerActivations(resnet18,'avgpool')
'''#conv_out.remove()
class layeractivation():
    features=None
    def __init__(self,module):
        self.hook=module.avgpool.register_forward_hook(self.hook_fn)
    def hook_fn(self,module,input,output):
        self.features=output.cpu().data.numpy()
    def remove(self):
        self.hook.remove()     
activation=layeractivation(resnet18)
for batch_idx , (data,target) in enumerate(dataloader):
    o = resnet18(data)
    print(batch_idx,activation.features)
activation.remove()
np.save('feature_pytorch.npy',activation.features) 
