import torch
import numpy as np
from fbnet import SuperNet
import re
from random import random

def extract():
    def is_theta(string):
        pattern = 'theta'
        if re.search(pattern, string) is not None:
            return 1
        else:
            return 0

    net = SuperNet(classes=17, batch_size=16)
    net.load_state_dict(torch.load('../model_save/supernet.pt'))

    layer_name_l = []
    params_l = []
    for name, param in net.named_parameters():
        if is_theta(name):
            # print(param.data)
            params_l.append(np.array(param.data))
            layer_name_l.append(name.split('.')[0])
    # print(layer_name_l)
    return params_l, layer_name_l

pm, ln = extract()
def sample(params, k=10):
    """
    这里需保证采样不重复（否则意义不大）(加入按概率最大方法采样的结果)
    :param params: 各层theta值
    :param k: 采样的组数（取几个神经网络）
    :return: 每个theta向量的取值（取第几个theta）（指示一组神经网络）  lists in list
    """
    theta_l = []
    def s1_sample(param):
        res_l = []
        for item in param:
            total = 0.
            for i in range(len(item[0])):
                total += item[0][i]

            r = random()
            idx = -1
            p = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(9):
                p[i] += item[0][i] / total
                addition = 0
                for j in range(i):
                    addition += item[0][j] / total
                p[i] += addition

            for i in range(9):
                if r < p[i]:
                    idx = i
                    break
            if idx == -1:
                idx = 8

            res_l.append(idx)
        return res_l

    def s2_sample():
        samp = s1_sample(params)
        bools = 0
        for item in theta_l:
            if item == samp:
                bools = 1
        return bools, samp

    # 这里将概率最大的网络添加进采样结果中
    temp = []
    for item in params:
        temp.append(np.array(item).argmax())
    theta_l.append(temp)

    for times in range(k):
        b = 1
        l = []
        while b:
            b, l = s2_sample()
        theta_l.append(l)

    return theta_l


