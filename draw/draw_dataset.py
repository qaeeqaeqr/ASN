import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def draw(data, name):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(10, 6))

    # colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    colors = plt.cm.Dark2(np.linspace(0, len(data), len(data)*len(data)))

    for i in range(len(data)):
        plt.bar(i, data[i], color=colors[(i*4)%len(data)], edgecolor='black', alpha=0.7)

    plt.xticks(range(len(data)),
               ['RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF',
                'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK'])
    plt.ylabel('Amount')
    plt.xlabel('Type of defects')
    plt.savefig('../outputs/'+name, dpi=300)


proxy_trainset = [817, 993, 800, 806, 1500, 245, 800, 586, 819, 847, 276, 806,
                  804, 800, 811, 800, 967]
proxy_testset = [160, 160, 135, 160, 160, 110, 116, 114, 160, 160, 36, 159, 152, 102,
                 160, 86, 160]
trainset = [45821, 184379, 16254, 19084, 283983, 6271, 22637, 23782, 74856,
            66499, 5010, 53986, 23685, 6746, 4625, 5325, 154624]
testset = [5538, 23624, 2021, 2038, 36218, 881, 2917, 2812, 9059, 7929,
           597, 6889, 3432, 765, 457, 612, 19655]

draw(proxy_trainset, 'proxyTrainset.pdf')
