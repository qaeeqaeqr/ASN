import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from fbnet import SuperNet
from mydataset import Train_dataset, Test_dataset


class Training_supernet(object):
    def __init__(self):
        self.classes = 17
        self.epoches = 9
        self.batch_size = 16
        self.lr = 0.01    # initial learning rate
        self.train_loader = DataLoader(Train_dataset(), batch_size=self.batch_size,
                                       shuffle=False, drop_last=True)    # 有了batch_size时，shuffle应设置为False
        self.test_loader = DataLoader(Test_dataset(), batch_size=1,
                                      shuffle=False, drop_last=False)
        self.save_path = '../model_save/supernet.pt'
        self.p_loss = []
        self.p_acc = []

    def train(self):
        net = SuperNet(classes=self.classes, batch_size=self.batch_size)
        if os.path.exists(self.save_path):         # model should be able to ba retrained
            net.load_state_dict((torch.load(self.save_path)))
            print('model loading...')
        loss_func = nn.BCELoss()
        optimizer = torch.optim.SGD(params=net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        for epoch in range(self.epoches):
            epoch_loss = 0.
            accuracy = Accuracy()

            for i, data in enumerate(self.train_loader):
                x, y_true = data
                y_predict = net(x)
                loss = loss_func(y_predict, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                accuracy(y_predict, torch.tensor(y_true, dtype=torch.int32))

            epoch_acc = accuracy.compute()

            self.p_loss.append(epoch_loss.detach().numpy())
            self.p_acc.append(epoch_acc)

            # info: epoch\ loss\ learning_rate\ acc
            print('------------------------------------------------------\n')
            print('epoch', epoch+1, ':')
            print('loss =', epoch_loss, '\t', 'with learning rate', self.lr)
            print('acc =', epoch_acc)

            accuracy.reset()
            if epoch % 2 == 1:
                self.lr *= 0.95


        torch.save(net.state_dict(), self.save_path)
        print('-->model has been saved')

    def draw(self):
        length = len(self.p_loss)
        x = []
        for i in range(length):
            x.append(i+1)

        plt.subplot(2, 1, 1)
        plt.plot(x, self.p_loss)
        plt.xlabel('epoches')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(x, self.p_acc)
        plt.xlabel('epoches')
        plt.ylabel('accuracy')

        plt.subplots_adjust(hspace=1.)
        plt.show()

training_supernet = Training_supernet()
training_supernet.train()
training_supernet.draw()