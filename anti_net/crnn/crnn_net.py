from telnetlib import SE
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

        # cnn3 layer
        self.cnn3 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal(self.cnn3.weight)
        self.bn3 = nn.BatchNorm2d(128)
        self.elu3 = nn.ELU(alpha=1.0, inplace=False)

        # pool layer
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

        return D, FM

class face_net_RNN(nn.Module):

    def __init__(self):
        super(face_net_RNN, self).__init__()

        self.device = torch.device("cuda")
        self.cnn = anti_spoof_net_CNN().to(self.device)
        self.hidden_dim = 100
        self.input_dim = 32*32
        self.num_layers = 1
        self.batch_szie = 1
        self.threshold = 0.1

        self.hidden = (torch.zeros(self.num_layers, self.batch_szie, self.hidden_dim).to(self.device),
                       torch.zeros(self.num_layers, self.batch_szie, self.hidden_dim).to(self.device))
        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, turned, anchors):
        D, FM = self.cnn(x)
        # F is [5,32,32,1]
        x = torch.ones(2, 1, 32, 32).to(self.device)
        y = torch.zeros(2, 1, 32, 32).to(self.device)
        V = torch.where(D >= self.threshold, x, y).to(self.device)
        U = FM * V
        if turned: # change face picture
            f = turn(U, anchors[i:i+1, :, :])
        else:
            f = U
        f = f.view(2, 1, -1)
        lstm_out, self.hidden = self.LSTM(f, self.hidden)
        x = self.fc(lstm_out)
        x = torch.fft(x, signal_ndim=1, normalized=False)
        return D, x  # X[2,1,2]


def resize_120(img):
    height, width, depth = img.shape
    res = np.resize((120, 120, depth), dtype=float)
    for x in range(120):
        realx = int(x*height/120)
        for y in range(120):
            realy = int(y*width/120)
            res[x,y,:] = img[realx, realy, :]
    return res

def resize_32(img):
    height, width, depth = img.shape
    dx, dy = int(height/32), int(width/32)
    res = np.resize((32,32,depth), dtype=float)
    for x in range(32):
        realx = x*dx
        for y in range(32):
            realy = y*dy
            res[x,y,:] = np.mean(np.mean(img[realx:realx+dx, realy:realy+dy, :],axis=0),axis=0)
    return res


def gen_offsets(kernel_size):
    offsets =np.zeros((2, kernel_size*kernel_size), dtype=np.int)
    ind = 0
    delta = (kernel_size -1)/2
    for i in range(kernel_size):
        y = i - delta
        for j in range(kernel_size):
            x = j - delta
            offsets[0, ind] = x
            offsets[1, ind] = y
            ind += 1
    return offsets


def rotate(img_base,anchor,kernel_size=3):
    img = resize_120(img_base)
    delta = (kernel_size-1)//2
    height, width, depth = img_base.shape
    res = np.zeros((64*kernel_size,64*kernel_size,depth), dtype=np.uint8)
    offsets = gen_offsets(kernel_size)
    for i in range(kernel_size*kernel_size):
        ox, oy = offsets[:, 1]
        index0 = anchor[0] + ox
        index1 = anchor[1] + oy
        temp = img[index1, index0, :].reshape(64, 64, depth).transpose(1,0,2)
        res[oy + delta::kernel_size, ox + delta::kernel_size, :] =temp
    return resize_32(res)



def turn(U, anchors, threshold=0.1):
    U_temp = np.cpu().array(U)
    height, width, depth = U_temp.shape
    f_temp = torch.zeros((32, 32, depth))
    for i in range(depth):
        f_temp[:, :, i:i+1] = rotate(U_temp[:,:,i:i+1], anchors[:,:,i])
    f = torch.from_numpy(f_temp)
    return f

