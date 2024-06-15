import shutil
from PIL import Image
from random import random
import torch
import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt

def make_proxy_trainset(picnum_in_cls=800, force_discard=1500):
    label_path = '../proxy_dataset/SewerML_Train.csv'
    tmp_names = os.listdir('G:/sewer_ML__dataset/')
    pattern = 'train'
    trains_folder = []
    for item in tmp_names:
        if re.search(pattern, item) is not None:
            trains_folder.append(item)

    # 从每个trainxx里选择一定数量的图片制作proxy_dataset
    # 一个里面选差不多2000张，总共26000张。   要注意类别均衡！！！
    statics = np.empty(18, dtype=np.int)
    f = open(label_path)
    reader = csv.reader(f)

    for item in trains_folder:
        folder_name = 'G:/sewer_ML__dataset/' + item + '/'
        img_names = os.listdir(folder_name)
        over = 0
        # 从当前文件夹中采样
        for pic in img_names:
            label = np.empty(17)
            for row in reader:
                if row[0] == pic:
                    for j in range(17):
                        label[j] = eval(row[3+j])
                    break
            """
            筛选标准：
            正常的只筛选两千张，十七类缺陷的图片每类数量为2000张
            过程：
            0、判定次图像为正常还是缺陷
            1、首先判定是否超量，若正常的超量，则直接弃用。若异常超量，执行以下流程
            2、若超量类为1的数量大于未超量类，则弃用；否则使用。
            3、当满足超量类大于15时，停止。
            """
            bools = 1
            for i in range(17):
                if label[i] != 0:
                    bools = 0
                    break

            if bools:
                # normal image
                if statics[17] < picnum_in_cls:
                    img = Image.open(folder_name+pic)
                    img.save('../proxy_dataset/proxy_train_set/'+pic.replace('0_2.png', '.jpg'))
                    img1 = Image.open('../proxy_dataset/proxy_train_set/'+pic.replace('0_2.png', '.jpg'))
                    new_img = img1.resize((224, 224), Image.ANTIALIAS)
                    new_img.save('../proxy_dataset/proxy_train_set/'+pic.replace('0_2.png', '.jpg'))
                    statics[17] += 1
                else:
                    pass
            else:
                greater = less = 0
                forced_pass = 0
                for j in range(17):
                    if label[j] == 1:
                        if statics[j] >= force_discard:
                            forced_pass = 1
                        elif statics[j] >= picnum_in_cls:
                            greater += 1
                        else:
                            less += 1
                if less > greater and not forced_pass:
                    img = Image.open(folder_name + pic)
                    img.save('../proxy_dataset/proxy_train_set/' + pic.replace('0_2.png', '.jpg'))
                    img1 = Image.open('../proxy_dataset/proxy_train_set/' + pic.replace('0_2.png', '.jpg'))
                    new_img = img1.resize((224, 224), Image.ANTIALIAS)
                    new_img.save('../proxy_dataset/proxy_train_set/' + pic.replace('0_2.png', '.jpg'))
                    for i in range(17):
                        statics[i] += label[i]
                else:
                    pass

            over = 0
            for i in range(18):
                if statics[i] >= picnum_in_cls:
                    over += 1
            if over > 15:
                break
        if over > 15:
            break
        print('folder', item, 'has been read')
    f.close()
    print('train-set finished')


def make_proxy_testset(total_pic_num=2000, max_picnum_in_cls=110, forced_discard=160):
    # 产生大约一万张测试数据，要考虑到缺陷类别的多样性，且其中的数据与训练集中的数据最好不要重复
    label_path = '../proxy_dataset/SewerML_Train.csv'
    tmp_names = os.listdir('G:/sewer_ML__dataset/')
    trains = os.listdir('../proxy_dataset/proxy_train_set/')
    pattern = 'train'
    tests_folder = []
    total = 0

    for item in tmp_names:
        if re.search(pattern, item) is not None:
            tests_folder.append(item)

    statics = np.empty(18, dtype=np.int)
    f = open(label_path)
    reader = csv.reader(f)
    for item in tests_folder:
        folder_name = 'G:/sewer_ML__dataset/' + item + '/'
        img_names = os.listdir(folder_name)
        for pic in img_names:
            if pic.replace('png', 'jpg') in trains:
                continue

            label = np.empty(17)
            for row in reader:
                if row[0] == pic:
                    for j in range(17):
                        label[j] = eval(row[3+j])
                    break
            bools = 1
            for i in range(17):
                if label[i] != 0:
                    bools = 0
                    break
            if bools:
                # normal image
                if statics[17] < max_picnum_in_cls:
                    img = Image.open(folder_name + pic)
                    img.save('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    img1 = Image.open('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    new_img = img1.resize((224, 224), Image.ANTIALIAS)
                    new_img.save('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    statics[17] += 1
                    total += 1
                else:
                    pass
            else:
                greater = less = 0
                forced_pass = 0
                for j in range(17):
                    if label[j] == 1:
                        if statics[j] >= forced_discard:
                            forced_pass = 1
                        elif statics[j] >= max_picnum_in_cls:
                            greater += 1
                        else:
                            less += 1
                if less >= greater and not forced_pass:
                    img = Image.open(folder_name + pic)
                    img.save('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    img1 = Image.open('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    new_img = img1.resize((224, 224), Image.ANTIALIAS)
                    new_img.save('../proxy_dataset/proxy_test_set/' + pic.replace('0_2.png', '.jpg'))
                    total += 1
                    for i in range(17):
                        statics[i] += label[i]
                else:
                    pass
            if total >= total_pic_num:
                break
        if total >= total_pic_num:
            break
        print('folder', item, 'has been read')
    f.close()
    print('test-set finished')


def show_proxy_trainset():
    train_path = 'D:/proxy_train_set/'
    label_path1 = '../proxy_dataset/SewerML_Train.csv'
    img_name_l = os.listdir(train_path)
    static = np.empty(17)

    f1 = open(label_path1)
    reader1 = csv.reader(f1)

    for item1 in img_name_l:
        item1 = item1.replace('jpg', 'png')
        for row1 in reader1:
            if row1[0] == item1:
                for k in range(17):
                    static[k] += eval(row1[3+k])
                break
    f1.close()
    print(static)
    plt.plot(static)
    plt.xlabel('defect')
    plt.ylabel('count')
    plt.title('proxy train-set')
    plt.show()

def show_proxy_testset():
    test_path = 'D:/proxy_test_set/'
    label_path1 = '../proxy_dataset/SewerML_Train.csv'
    img_name_l = os.listdir(test_path)
    static = np.empty(17)

    f1 = open(label_path1)
    reader1 = csv.reader(f1)

    for item1 in img_name_l:
        item1 = item1.replace('jpg', 'png')
        for row1 in reader1:
            if row1[0] == item1:
                for k in range(17):
                    static[k] += eval(row1[3 + k])
                break
    f1.close()
    print(static)
    plt.plot(static)
    plt.xlabel('defect')
    plt.ylabel('count')
    plt.title('proxy test-set')
    plt.show()

show_proxy_trainset()
show_proxy_testset()