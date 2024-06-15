import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sampled_nn import *
from mydataset import Sampled_test_dataset, Sampled_train_dataset
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchmetrics import Accuracy
import torchmetrics


class Training_sampled_n(object):
    def __init__(self, n, epochs=2, lr=6e-4):      # n means train the nth subnet
        """

        :param n: appoint a subnet to train(0-6)
        """
        self.epoches = epochs
        self.batch_size = 512
        self.lr = lr
        self.classes = 17
        self.n = n
        self.train_loader = DataLoader(Sampled_train_dataset(), batch_size=self.batch_size,
                                       shuffle=True, num_workers=16,
                                       drop_last=True, prefetch_factor=5)
        self.test_loader = DataLoader(Sampled_test_dataset(), batch_size=self.batch_size,
                                      shuffle=False, num_workers=16,
                                      drop_last=False, prefetch_factor=5)
        self.save_path = './model_save/sampled_net' + str(self.n) + '.pt'
        self.p_loss = []
        self.p_acc = []

    def train(self):
        if self.n == 0:
            net = Sampled_nn0(batch_size=self.batch_size)
        elif self.n == 1:
            net = Sampled_nn1(batch_size=self.batch_size)
        elif self.n == 2:
            net = Sampled_nn2(batch_size=self.batch_size)
        elif self.n == 3:
            net = Sampled_nn3(batch_size=self.batch_size)
        elif self.n == 4:
            net = Sampled_nn4(batch_size=self.batch_size)
        elif self.n == 5:
            net = Sampled_nn5(batch_size=self.batch_size)
        else:
            net = Sampled_nn6(batch_size=self.batch_size)


        if torch.cuda.is_available():
            print('GPU can be used.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device using:', device)
        torch.cuda.empty_cache()
        net.to(device)
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
            print('model loading')

        weight = np.empty((self.batch_size, self.classes), dtype=np.float16)
        weight_l = [21.77, 5.41, 61.37, 52.27, 3.51, 159.08, 44.07, 41.95, 13.33, 15.00, 199.12, 18.48,
                    42.12, 147.88, 215.69, 187.34, 6.45]
        weight_n = [1 for i in range(17)]
        for i in range(17):
            weight[:, i] = weight_l[i]
        loss_func = nn.BCELoss(weight=torch.from_numpy(weight).to(device))
        optimizer = optim.SGD(params=net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        print('start training!\n')
        for epoch in range(self.epoches):
            total_loss = 0.
            accuracy = Accuracy(task="multilabel", num_labels=17)
            count = 0

            for i, data in enumerate(self.train_loader):
                x1, y_true1 = data
                x, y_true = x1.to(device), y_true1.to(device)
                y_pred = net(x)
                print(y_ture)
                print(y_true.shape)
                loss = loss_func(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

                if count % 1 == 0:
                    print('\r', 'epoch', epoch, ', step', count, ':', 'temp_loss =', total_loss, end='')

            epoch_acc = accuracy.compute()

            self.p_loss.append(total_loss.cpu().detach().numpy())
            self.p_acc.append(epoch_acc)

            print('\n------------------------------------------------------\n')
            print('epoch', epoch, ':')
            print('train_loss =', total_loss.cpu().detach().numpy(), '\t', 'with learning rate', self.lr)
            print('train_acc =', epoch_acc)
            print('\n------------------------------------------------------\n')

            accuracy.reset()

            torch.cuda.empty_cache()

        torch.save(net.state_dict(), self.save_path)
        print('model saved')

        f_log = open('log.txt', 'w')
        f_log.write('第一行为loss，第二行为acc。')
        f_log.write(self.p_loss)
        f_log.write(self.p_acc)
        f_log.close()
        print('recorded!')

    def test(self):
        if self.n == 0:
            net = Sampled_nn0(batch_size=self.batch_size)
        elif self.n == 1:
            net = Sampled_nn1(batch_size=self.batch_size)
        elif self.n == 2:
            net = Sampled_nn2(batch_size=self.batch_size)
        elif self.n == 3:
            net = Sampled_nn3(batch_size=self.batch_size)
        elif self.n == 4:
            net = Sampled_nn4(batch_size=self.batch_size)
        elif self.n == 5:
            net = Sampled_nn5(batch_size=self.batch_size)
        else:
            net = Sampled_nn6(batch_size=self.batch_size)

        save_path = self.save_path
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
            print('model loaded')
        test_loader = DataLoader(Sampled_test_dataset(), batch_size=64, shuffle=False, drop_last=True)
        loss_func = nn.BCELoss()
        epoch_test_loss = 0.
        epoch_test_acc = 0.

        print('evaluating...')
        count = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                y_predict = net(x)
                loss = loss_func(y_predict, y_true)
                epoch_test_loss += loss
                acc = torchmetrics.functional.accuracy(y_predict, torch.tensor(y_true, dtype=torch.int32))
                epoch_test_acc += acc.detach().numpy()
                count += 1
                del x, loss, acc, y_true, y_predict
        print('\n---------------------------------------------------------')
        print('val loss: ', epoch_test_loss)
        print('val acc: ', epoch_test_acc / count)
        print('---------------------------------------------------------\n')

    def draw(self):
        length = len(self.p_loss)
        print('total_epoch: ', length)
        x = []
        for i in range(length):
            x.append(i + 1)
        plt.subplot(2, 1, 1)
        plt.plot(x, self.p_loss)
        plt.xlabel('epoches')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(x, self.p_acc)
        plt.xlabel('epoches')
        plt.ylabel('accuracy')

        plt.subplots_adjust(hspace=0.8)
        plt.show()

training_sampled0 = Training_sampled_n(0)
# test per n epochs
training_sampled0.train()
training_sampled0.test()
