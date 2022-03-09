import torch
import torch.nn as nn
import numpy as np
import torchvision


class face_net_RNN(nn.Module):

    def __init__(self):
        super(face_net_RNN, self).__init__()

        self.device = torch.device("cuda")
        self.i = 0

        self.hidden_dim = 100
        self.input_dim = 32*32
        self.num_layers = 1
        self.batch_szie = 1
        self.threshold = 0.1

        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers).to(self.device)

        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, f):
        # F is [10,32,32,1]
        f = f.view(10, 1, -1)
        if self.i == 0:
            self.hidden = (torch.zeros(self.num_layers, self.batch_szie, self.hidden_dim).to(self.device),
                           torch.zeros(self.num_layers, self.batch_szie, self.hidden_dim).to(self.device))
        else:
            print('not_init')
            self.i += 1
        # print("hidden:", self.hidden)
        lstm_out, self.hidden = self.LSTM(f, self.hidden)
        x = self.fc(lstm_out)
        # print("x:", x)
        x = torch.fft(x, signal_ndim=1, normalized=False)
        # print("x_fft:", x)
        return x # X[10,1,2]


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