from searching_layers import TBS_single_layer
from blocks import *

import torch
from torch import nn
import torch.nn.functional as f

class Sampled_nn0(nn.Module):
    def __init__(self, batch_size):
        super(Sampled_nn0, self).__init__()
        self.classes = 17
        self.batch_size = batch_size

        # 从文件中读取采样结果
        self.pm = []
        f1 = open('../sample_result.txt', 'r')
        for i in range(6+1):
            insert_lst = []
            tmp_str = f1.readline()
            tmp_lst = tmp_str.split(' ')
            tmp_lst.remove('\n')
            for item in tmp_lst:
                insert_lst.append(eval(item))
            self.pm.append(insert_lst)
        print(self.pm)
        f1.close()

        self.result = self.pm[0]        # 抽样了多组神经网络，这里先取其中的一组进行实验（重新训练）
        self.blk_order_list = ['Block_k3_e1', 'Block_k3_e1_g2', 'Block_k3_e3', 'Block_k3_e6',    # 依次对应order中的0-8的序号
                               'Block_k5_e1', 'Block_k5_e1_g2', 'Block_k5_e3', 'Block_k5_e6', 'Block_skip']
        self.in_channel_list = [16, 16, 16, 24, 24, 24, 32, 32, 32, 64, 64, 64,
                                112, 112, 112, 184, 184]
        self.out_channel_list = [16, 16, 24, 24, 24, 32, 32, 32, 64, 64, 64, 112,
                                 112, 112, 184, 184, 352]
        self.stride_list = [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=2, padding=1)  # to 112*112*16

        self.blk0 = eval(self.blk_order_list[self.result[0]])(in_channel=self.in_channel_list[0],
                                                                out_channel=self.out_channel_list[0],
                                                                stride=self.stride_list[0])
        self.blk1 = eval(self.blk_order_list[self.result[1]])(in_channel=self.in_channel_list[1],
                                                              out_channel=self.out_channel_list[1],
                                                              stride=self.stride_list[1])
        self.blk2 = eval(self.blk_order_list[self.result[2]])(in_channel=self.in_channel_list[2],
                                                              out_channel=self.out_channel_list[2],
                                                              stride=self.stride_list[2])
        self.blk3 = eval(self.blk_order_list[self.result[3]])(in_channel=self.in_channel_list[3],
                                                              out_channel=self.out_channel_list[3],
                                                              stride=self.stride_list[3])
        self.blk4 = eval(self.blk_order_list[self.result[4]])(in_channel=self.in_channel_list[4],
                                                              out_channel=self.out_channel_list[4],
                                                              stride=self.stride_list[4])
        self.blk5 = eval(self.blk_order_list[self.result[5]])(in_channel=self.in_channel_list[5],
                                                              out_channel=self.out_channel_list[5],
                                                              stride=self.stride_list[5])
        self.blk6 = eval(self.blk_order_list[self.result[6]])(in_channel=self.in_channel_list[6],
                                                              out_channel=self.out_channel_list[6],
                                                              stride=self.stride_list[6])
        self.blk7 = eval(self.blk_order_list[self.result[7]])(in_channel=self.in_channel_list[7],
                                                              out_channel=self.out_channel_list[7],
                                                              stride=self.stride_list[7])
        self.blk8 = eval(self.blk_order_list[self.result[8]])(in_channel=self.in_channel_list[8],
                                                              out_channel=self.out_channel_list[8],
                                                              stride=self.stride_list[8])
        self.blk9 = eval(self.blk_order_list[self.result[9]])(in_channel=self.in_channel_list[9],
                                                              out_channel=self.out_channel_list[9],
                                                              stride=self.stride_list[9])
        self.blk10 = eval(self.blk_order_list[self.result[10]])(in_channel=self.in_channel_list[10],
                                                              out_channel=self.out_channel_list[10],
                                                              stride=self.stride_list[10])
        self.blk11 = eval(self.blk_order_list[self.result[11]])(in_channel=self.in_channel_list[11],
                                                              out_channel=self.out_channel_list[11],
                                                              stride=self.stride_list[11])
        self.blk12 = eval(self.blk_order_list[self.result[12]])(in_channel=self.in_channel_list[12],
                                                              out_channel=self.out_channel_list[12],
                                                              stride=self.stride_list[12])
        self.blk13 = eval(self.blk_order_list[self.result[13]])(in_channel=self.in_channel_list[13],
                                                              out_channel=self.out_channel_list[13],
                                                              stride=self.stride_list[13])
        self.blk14 = eval(self.blk_order_list[self.result[14]])(in_channel=self.in_channel_list[14],
                                                              out_channel=self.out_channel_list[14],
                                                              stride=self.stride_list[14])
        self.blk15 = eval(self.blk_order_list[self.result[15]])(in_channel=self.in_channel_list[15],
                                                              out_channel=self.out_channel_list[15],
                                                              stride=self.stride_list[15])
        self.blk16 = eval(self.blk_order_list[self.result[16]])(in_channel=self.in_channel_list[16],
                                                              out_channel=self.out_channel_list[16],
                                                              stride=self.stride_list[16])

        self.conv2 = nn.Conv2d(in_channels=352, out_channels=1504, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(in_features=1504, out_features=512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=self.classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)
        x = self.blk9(x)
        x = self.blk10(x)
        x = self.blk11(x)
        x = self.blk12(x)
        x = self.blk13(x)
        x = self.blk14(x)
        x = self.blk15(x)
        x = self.blk16(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(self.batch_size, -1)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

class Sampled_nn1(Sampled_nn0):
    def __init__(self, batch_size):
        super(Sampled_nn1, self).__init__(batch_size=batch_size)
        self.result = self.pm[1]

class Sampled_nn2(Sampled_nn0):
    def __init__(self, batch_size):
        super(Sampled_nn2, self).__init__(batch_size=batch_size)
        self.result = self.pm[2]

class Sampled_nn3(Sampled_nn0):
    def __init__(self, batch_size):
        super(Sampled_nn3, self).__init__(batch_size=batch_size)
        self.result = self.pm[3]



