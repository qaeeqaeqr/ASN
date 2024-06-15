import torch
import torch.nn.functional as f
from torch import nn
from searching_layers import TBS_single_layer
from blocks import *


class SuperNet(nn.Module):     # with input shape 224*224*3
    def __init__(self, classes, batch_size):
        torch.autograd.set_detect_anomaly(True)
        super(SuperNet, self).__init__()
        self.classes = classes
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=2, padding=1)   # to 112*112*16
        self.tbslayer1_n1 = TBS_single_layer(in_channel=16, out_channel=16, stride=1)   # to 112*112*16
        self.tbslayer2_n1 = TBS_single_layer(in_channel=16, out_channel=16, stride=2)
        self.tbslayer2_n2 = TBS_single_layer(in_channel=16, out_channel=24, stride=1)
        self.tbslayer2_n3 = TBS_single_layer(in_channel=24, out_channel=24, stride=1)  # to 56*56*24
        self.tbslayer3_n1 = TBS_single_layer(in_channel=24, out_channel=24, stride=2)
        self.tbslayer3_n2 = TBS_single_layer(in_channel=24, out_channel=32, stride=1)
        self.tbslayer3_n3 = TBS_single_layer(in_channel=32, out_channel=32, stride=1)   # to 28*28*32
        self.tbslayer4_n1 = TBS_single_layer(in_channel=32, out_channel=32, stride=2)
        self.tbslayer4_n2 = TBS_single_layer(in_channel=32, out_channel=64, stride=1)
        self.tbslayer4_n3 = TBS_single_layer(in_channel=64, out_channel=64, stride=1)  # to 14*14*64
        self.tbslayer5_n1 = TBS_single_layer(in_channel=64, out_channel=64, stride=1)
        self.tbslayer5_n2 = TBS_single_layer(in_channel=64, out_channel=112, stride=1)
        self.tbslayer5_n3 = TBS_single_layer(in_channel=112, out_channel=112, stride=1)  # to 14*14*112
        self.tbslayer6_n1 = TBS_single_layer(in_channel=112, out_channel=112, stride=2)
        self.tbslayer6_n2 = TBS_single_layer(in_channel=112, out_channel=184, stride=1)
        self.tbslayer6_n3 = TBS_single_layer(in_channel=184, out_channel=184, stride=1)  # to 7*7*184
        self.tbslayer7_n1 = TBS_single_layer(in_channel=184, out_channel=352, stride=1)    # to 7*7*352
        self.conv2 = nn.Conv2d(in_channels=352, out_channels=1504, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(in_features=1504, out_features=512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tbslayer1_n1(x)
        x = self.tbslayer2_n1(x)
        x = self.tbslayer2_n2(x)
        x = self.tbslayer2_n3(x)
        x = self.tbslayer3_n1(x)
        x = self.tbslayer3_n2(x)
        x = self.tbslayer3_n3(x)
        x = self.tbslayer4_n1(x)
        x = self.tbslayer4_n2(x)
        x = self.tbslayer4_n3(x)
        x = self.tbslayer5_n1(x)
        x = self.tbslayer5_n2(x)
        x = self.tbslayer5_n3(x)
        x = self.tbslayer6_n1(x)
        x = self.tbslayer6_n2(x)
        x = self.tbslayer6_n3(x)
        x = self.tbslayer7_n1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(self.batch_size, -1)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        return f.sigmoid(self.fc2(x))

