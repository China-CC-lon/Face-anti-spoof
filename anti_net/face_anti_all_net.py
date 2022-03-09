import torch
import numpy as np
import torch.nn as nn

from anti_net.crnn import face_anti_RNN, face_anti_CNN


class face_anti_all_net(nn.Module):
    def __init__(self) :
        super(face_anti_all_net, self).__init__()

        self.device = torch.device("cuda")

        self.CNN = face_anti_CNN.anti_spoof_net_CNN().to(self.device)
        # self.CNN.load_state_dict(torch.load('face_anti_model_cnn.pth'))
        self.RNN = face_anti_RNN.face_net_RNN().to(device=self.device)

        self.threshold = 0.1

    def forward(self, image):
        D, FM = self.CNN(image)  # image [10, 3, 256, 256]
        x = torch.ones(10, 1, 32, 32).to(self.device)  # x[10, 1, 32, 32]
        y = torch.zeros(10, 1, 32, 32).to(self.device)  # y[10, 1, 32, 32]
        V = torch.where(D >= self.threshold, x, y).to(self.device)
        U = FM * V  # U [10, 1, 32, 32]
        # F = turn(U, anchors)  # anchors[10, 2, 4096]
        R = self.RNN(U)
        return D, R


def resize_120(img):
    depth, _, height, width= img.shape
    res = np.resize((120, 120, depth))
    for x in range(120):
        realx = int(x*height/120)
        for y in range(120):
            realy = int(y*width/120)
            res[x, y, :] = img[realx, realy, :]
    return res


def resize_32(img):
    depth, _, height, width = img.shape
    dx, dy = int(height/32), int(width/32)
    res = np.resize((32, 32, depth), dtype=float)
    for x in range(32):
        realx = x*dx
        for y in range(32):
            realy = y*dy
            res[x, y, :] = np.mean(np.mean(img[realx:realx+dx, realy:realy+dy, :], axis=0), axis=0)
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


def rotate(img_base, anchor, kernel_size=3):
    # img_base[10,1,32,32]
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



def turn(U, anchors):
    U = U.cpu().detach().numpy()
    U_temp = np.array(U)  # [10, 1, 32, 32]
    # print(U_temp.shape)
    depth, _, height, width = U_temp.shape
    f_temp = torch.zeros((depth, 32, 32))  # [10, 32, 32]
    for i in range(depth):  # depth = 10
        f_temp[i:i+1, :, :] = rotate(U_temp[i:i+1, :, :], anchors[i, :, :])
    f = torch.from_numpy(f_temp)
    return f
