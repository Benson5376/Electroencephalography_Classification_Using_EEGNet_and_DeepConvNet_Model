import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet,self).__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(), #1.0
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25) #0.25
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(), #1.0
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25) #0.25
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x=self.firstconv(x)
        x=self.depthwiseConv(x)
        x=self.separableConv(x)
        x = x.view((-1 , 736))
        x=self.classify(x)
        return x

class DeepConvNet(nn.Module):
    
    def __init__(self):
        super(DeepConvNet,self).__init__()
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.conv3 = nn.Sequential(
            
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        self.conv4 = nn.Sequential(
            
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(1, 2),
            nn.Dropout(p=0.5)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )
        
        
    def forward(self, x):
            
        x = self.conv1(x)
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = self.conv4(x)
        x = x.view(-1, 8600)
        x = self.classify(x)
            
        return x

    
class MLP(nn.Module):
    
    def __init__(self):
        super(MLP,self).__init__()
        
        self.linear1=torch.nn.Linear(1500, 512)
        self.relu = torch.nn.LeakyReLU()
        
        self.linear2=torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.LeakyReLU()
        
        self.linear3=torch.nn.Linear(256, 2)
        self.relu3 = torch.nn.LeakyReLU()
    
    def forward(self, x):
        x=self.linear1(x)
        x=self.relu(x)
        
        x=self.linear2(x)
        x=self.relu2(x)
        
        x=self.linear3(x)
        x=self.relu3(x)
        
        return x
    