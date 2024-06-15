import torch
from torch import nn
import torch.nn.functional as f

class Block_k3_e1(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k3_e1, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=3, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k3_e1_g2(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k3_e1_g2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=2)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=3, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=2)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k3_e3(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k3_e3, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 3
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=3, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k3_e6(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k3_e6, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 6
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=3, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k5_e1(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k5_e1, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=5, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k5_e1_g2(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k5_e1_g2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=2)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=5, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=2)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k5_e3(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k5_e3, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 3
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=5, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_k5_e6(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_k5_e6, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.expansion = 6
        self.conv1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=self.stride, kernel_size=1, padding=0, groups=1)
        self.bn1 = nn.BatchNorm2d(self.expansion * self.in_channel)
        self.conv2 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.expansion * self.in_channel,
                               stride=1, kernel_size=5, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = nn.Conv2d(in_channels=self.expansion * self.in_channel, out_channels=self.out_channel,
                               stride=1, kernel_size=1, padding='same', groups=1)
        self.conv_for_dimkeep = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                                          stride=self.stride, kernel_size=1)

    def forward(self, x):
        x_cpy = x
        x_cpy = self.conv_for_dimkeep(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.add(x, x_cpy)


class Block_skip(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Block_skip, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                              stride=1, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        if self.in_channel==self.out_channel and self.stride==1:
            return x
        if self.in_channel==self.out_channel and self.stride==2:
            return self.pool(x)
        if self.in_channel!=self.out_channel and self.stride==1:
            return self.bn(self.conv(x))