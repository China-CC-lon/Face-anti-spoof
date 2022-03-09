import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from anti_net import face_anti_all_net
from anti_net.crnn import face_anti_CNN


# picture show
def imshow_np(img):
    h, w, d = img.shape
    print(img.shape)
    if d == 1:
        img = img[:, :, 0]
    # cv2.imshow('figure',img)
    plt.imshow(img)
    plt.show()


def imshow(img):
    imshow_np(img.numpy())


class data_perpare:
    def __init__(self) -> None:
        self.n = None
        self.Images = None
        self.Anchors = None
        self.Label_D = None
        self.Labels = None
        self.data_images_train =None
        self.data_anchors_train = None
        self.data_labels_d_train = None
        self.data_labels_train = None
        self.data_images_test = None
        self.data_anchors_test = None
        self.data_labels_d_test = None
        self.data_labels_test = None

    # load data from file
    def load_file(self, i_f, a_f, l_D_f, l_f):
        self.Images = np.load(i_f) # image data
        self.Anchors = np.load(a_f) # feature point
        self.Label_D = np.load(l_D_f) # the labels of deep map picture
        self.Labels = np.load(l_f)


        self.n = len(self.Images) # the number of image
        self.set_data()
        trainloader_D, testloader_D = self.set_data_torch_container()
        return trainloader_D, testloader_D

    def set_data(self):
        n = self.n
        data_images = np.zeros((n, 256, 256, 3), dtype=np.float32)
        data_anchors = np.zeros((n, 2, 4096), dtype=np.float32)
        data_labels_d = np.zeros((n, 32, 32, 1), dtype=np.float32)
        data_labels = np.zeros((n), dtype=np.float32)
        print(data_labels.shape)
        
        for item in self.Images.files:
            data_images[int(item), :, :] = self.Images[item]
            data_anchors[int(item), :, :] = self.Anchors[item]
            data_labels_d[int(item), :, :] = self.Label_D[item]
            data_labels[int(item)] = self.Labels[item]
        
        training_part = 2/5
        train_n = int(n * training_part)

        # train date set
        self.data_images_train = data_images[:train_n, :, :]  # [2400, 256, 256, 3]
        self.data_anchors_train = data_anchors[:train_n, :, :]  # [2400, 2, 4096]
        self.data_labels_d_train = data_labels_d[:train_n, :, :]  # [2400, 32, 32, 1]
        self.data_labels_train = data_labels[:train_n]  # [2400,]


        # test data set
        self.data_images_test = data_images[train_n:, :, :]
        self.data_anchors_test = data_anchors[train_n:, :, :]
        self.data_labels_d_test = data_labels_d[train_n:, :, :]
        self.data_labels_test = data_labels[train_n:]

    def set_data_torch_container(self):
        trainset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(self.data_images_train, (0, 3, 1, 2))), torch.tensor(self.data_labels_d_train))
        testset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(self.data_images_test, (0, 3, 1, 2))), torch.tensor(self.data_labels_d_test))

        trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=10, shuffle=True)
        testloader_D = torch.utils.data.DataLoader(testset_D, batch_size=10, shuffle=True)


        return trainloader_D, testloader_D


class face_anti:
    def __init__(self):
        pass
    
    def train_CNN(self, net, optimizer, trainloader, anchors, criterion, n_epoch=10):
        print("train_CNN训练开始")
        total = 0
        sel = True
        loss_list_cnn = []
        for epoch in range(n_epoch):
            print("epoch:{}".format(epoch))
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                torch.cuda.empty_cache()
                images, labels_D = data
                # print(images[0, :, :, :].shape)
                images, labels_D = images.to(device), labels_D.to(device)

                # training step
                optimizer.zero_grad()
                out_D, _ = net(images)

                # handle NaN
                if (torch.norm((out_D != out_D).float())==0):

                    if(i%10==0):
                        imshow_np(np.transpose(images[1, :, :, :].data.cpu().numpy(), (1, 2, 0))/255)
                        imshow_np(np.transpose(labels_D[1, :, :, :].data.cpu().detach().numpy(), (1, 0, 2)))
                        imshow_np(np.transpose(out_D[1, :, :, :].data.cpu().detach().numpy(), (1, 2, 0)))

                    loss = criterion(out_D, labels_D)
                    loss.backward()
                    optimizer.step()

                    # compute statistics
                    total += labels_D.size(0)
                    running_loss += loss.item()
                    torch.cuda.empty_cache()
                    print('[%d,%5d] loss:%.3f' % (epoch+1, i+1, running_loss/total))
            print('Epoch finished')
            loss_list_cnn.append(running_loss)
            # torch.save(net, 'face_anti_model1.pth')
            # torch.save(net.state_dict(), 'face_anti_param1.pth')
        print('Finished Training CNN')

    def train_RNN(self, net, optimizer, trainloader, anchors, labels, criterion, n_epoch=10):
        print("RNN训练开始!")
        total = 0
        sel = False
        loss_list_rnn = []
        for epoch in range(n_epoch):
            print("epoch:{}".format(epoch))
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                print(i)
                torch.cuda.empty_cache()
                images, labels_D = data
                images, labels_D = images.to(device), labels_D.to(device)

                # training step
                optimizer.zero_grad()  # 清除梯度
                _, out_F = net(images)  # anchors[10,2,4096]

                if (torch.norm((out_F != out_F).float())==0):

                    if (i%50==0 or i%50==1):
                        print('F:')
                        print(out_F)

                    if labels[i*10] == 0:
                        label = torch.zeros((10, 1, 2), dtype=torch.float32)
                        label = label.to(device)
                    else:
                        label = torch.ones((10, 1, 2), dtype=torch.float32)
                        label = label.to(device)

                    loss = criterion(out_F, label)
                    # loss_list_rnn.append(loss)
                    loss.backward()
                    optimizer.step()

                    total += labels_D.size(0)
                    running_loss += loss.item()
                    torch.cuda.empty_cache() # 释放gpu内存
                    print('[%d, %5d] loss:%.3f' % (epoch+1, i+1, running_loss/total))
            print('Epoch finished')
            loss_list_rnn.append(running_loss)
            # torch.save(net, 'face_anti_model2.pth')
            # torch.save(net.state_dict(), 'face_anti_model_param2.pth')
        print('Finished Training RNN')
        self.show_loss_curve(loss_list_rnn, 'CRNN_loss_curve')

    def show_loss_curve(self, loss, name):
        l = len(loss)
        x = []
        for i in range(l):
            x.append(i+1)
        plt.title(name)
        plt.plot(x, loss, 'b', label='loss')
        plt.xlabel('num_epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()


    def train(self, net, optimizer, trainloader, data_anchors, data_labels, criterion, n_epoch):
        print("训练开始!")
        self.train_CNN(net, optimizer, trainloader, data_anchors, criterion, n_epoch=n_epoch)
        torch.save(net.state_dict(), 'face_anti_model_cnn1.pth')
        # self.show_loss_curve(loss_cnn, 'CNN_loss_curve')

        self.train_RNN(net, optimizer, trainloader, data_anchors, data_labels, criterion, n_epoch=40)
        print("CRNN整体训练结束!")
        # self.show_loss_curve(loss_rnn, 'CRNN_loss_curve')


    def test(self, net, testloader, data_anchors_test, labels):
        correct = 0
        total = 0
        # l = 0.015
        with torch.no_grad():
            for i, (images, _) in enumerate(testloader, 0):
                images = images.to(device)
                label = labels[i*10]
                print(images.shape)
                out_D, out_F = net(images) # D[10, 1, 32, 32] # [10,1,2]
                print("out_F, out_D:", out_F, out_D)
                value = 0.015*torch.norm(out_D).pow(2)+torch.norm(out_F).pow(2)
                print("value:", value)
                if (value>1644 and label==1):
                    correct += 1
                if (value<1644 and label==0):
                    correct += 1
                total += 1
                print(correct/total)
            accuracy = correct / total
        # loss = loss / total
        return accuracy


if __name__ == "__main__": 
    all_n_epoch = 10
    device = torch.device("cuda")
    dataset = data_perpare()

    i_f = 'M:/xilinx_comtest/face-anit/data_processing/images_5.npz'
    a_f = 'M:/xilinx_comtest/face-anit/data_processing/anchors_5.npz'
    l_D_f = 'M:/xilinx_comtest/face-anit/data_processing/labels_D_5.npz'
    l_f = 'M:/xilinx_comtest/face-anit/data_processing/label_5.npz'

    trainloader_D, testloader_D = dataset.load_file(i_f, a_f, l_D_f, l_f)
    print(trainloader_D)
    data_anchors_train, data_labels_train, data_labels_test = dataset.data_anchors_train, dataset.data_labels_train, dataset.data_labels_test
    data_anchors_test = dataset.data_anchors_test

    model_train = face_anti_all_net.face_anti_all_net().to(device)
    # net_CNN = face_anti_CNN.anti_spoof_net_CNN().to(device)
    criterion = nn.MSELoss()

    opt = optim.Adam(model_train.parameters(), lr=3e-3, betas=(0.9, 0.99), eps=1e-08)
    anti_face = face_anti()

    # train
    anti_face.train(model_train, opt, trainloader_D,
                    data_anchors_train, data_labels_train, criterion, n_epoch=200)

    mon_model_test = face_anti_all_net.face_anti_all_net().to(device)
    mon_model_test.load_state_dict(torch.load("face_anti_model_param2.pth"))
    mon_model_test.eval()
    # mon_model_test = torch.load('face_anti_model.pth')

    # test
    accuary = anti_face.test(mon_model_test, testloader_D, data_anchors_test, data_labels_test)
    print('正确率: %.3f' % (accuary*100))

    







