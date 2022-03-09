from tkinter import N
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torchvision

class block(nn.Module):
    
    def __init__(self, begin):
        super(block, self).__init__()
        
        # cnn1 layer
        if begin == True:
            self.cnn1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        else:
            self.cnn1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal(self.cnn1.weight)
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU(alpha=1.0, inplace=False)
        
        # cnn2 layer
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal(self.cnn2.weight)
        self.bn2 = nn.BatchNorm2d(196)
        self.elu2 = nn.ELU(alpha=1.0, inplace=False)

        #cnn3 layer
        self.cnn3 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal(self.cnn3.weight)
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU(alpha=1.0, inplace=False)

        #pool layer
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    # forward
    def forward(self, x):
        x = self.elu1(self.bn1(self.cnn1(x)))
        x = self.elu2(self.bn2(self.cnn2(x)))
        x = self.elu3(self.bn3(self.cnn3(x)))
        x = self.pool(x)
        return x

class anti_spoof_net_CNN(nn.Module):
    
    def __init__(self):
        super(anti_spoof_net_CNN, self).__init__()

        self.resize_32 = nn.Upsample(size=32, mode='nearest')
        self.resize_64 = nn.Upsample(size=64, mode='nearest')

        self.cnn0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal(self.cnn0.weight)
        self.bn0 = nn.BatchNorm2d(64)
        self.elu0 = nn.ELU(alpha=1.0, inplace=False)

        self.block1 = block(True)
        self.block2 = block(False)
        self.block3 = block(False)

        # feature map
        self.cnn4 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # deep map
        self.cnn7 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.cnn9 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.elu0(self.bn0(self.cnn0(x)))
        
        # block1
        x = self.block1(x)
        X1 = self.resize_64(x)

        # block2
        x = self.block2(x)
        X2 = self.resize_64(x)

        # block3
        x = self.block3(x)
        X3 = self.resize_64(x)

        X = torch.cat((X1, X2, X3), 1)

        # feature
        FM = self.cnn4(X)
        FM = self.cnn5(FM)
        FM = self.cnn6(FM)
        FM = self.resize_32(FM)

        # deep
        D = self.cnn7(X)
        D = self.cnn8(D)
        D = self.cnn9(D)
        D = self.resize_32(D)

        return D, FM  # D [10, 1, 32, 32] FM [10, 1, 32, 32]



