from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.models import load_model, save_model
from tensorflow.python.keras import initializers
from tensorflow.python.keras import metrics
import tensorflow.python.keras.backend as K
from sklearn import metrics as skmetric

import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import csv
import random


class Grouped_conv2d(layers.Layer):
    # 这里只根据自己的需求，完成分两组的卷积即可
    # 即卷积组数只能为2
    def __init__(self, filters, kernel_size=1, strides=(1, 1), padding='valid', groups=2):
        super(Grouped_conv2d, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups

        # prepare for network
        self.conv1 = layers.Conv2D(filters=int(self.filters / 2), kernel_size=self.kernel_size,
                                   strides=self.strides, padding=self.padding)
        self.conv2 = layers.Conv2D(filters=int(self.filters / 2), kernel_size=self.kernel_size,
                                   strides=self.strides, padding=self.padding)

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        channels = x.shape[-1]
        x_split1 = x[:, :, :, :int(channels // 2)]
        x_split2 = x[:, :, :, int(channels // 2):]

        x_split1 = self.conv1(x_split1)
        x_split2 = self.conv2(x_split2)

        return tf.concat([x_split1, x_split2], axis=3)


class K3_e1(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K3_e1, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 1

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2d(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K3_e1_g2(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K3_e1_g2, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 1

        self.conv1 = Grouped_conv2d(filters=self.in_channel * self.expansion, kernel_size=1,
                                    strides=(self.stride, self.stride), padding='valid', groups=2)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same')
        self.conv3 = Grouped_conv2d(filters=self.out_channel, kernel_size=1,
                                    strides=(1, 1), padding='same', groups=2)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K3_e3(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K3_e3, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 3

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K3_e6(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K3_e6, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 6

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K5_e1(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K5_e1, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 1

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K5_e1_g2(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K5_e1_g2, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 1

        self.conv1 = Grouped_conv2d(filters=self.in_channel * self.expansion, kernel_size=1,
                                    strides=(self.stride, self.stride), padding='valid', groups=2)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same')
        self.conv3 = Grouped_conv2d(filters=self.out_channel, kernel_size=1,
                                    strides=(1, 1), padding='same', groups=2)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K5_e3(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K5_e3, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 3

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class K5_e6(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K5_e6, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 6

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)  # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same')
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride)

    def call(self, x, **kwargs):
        x_cpy = x
        x_cpy = self.dimkeep_conv(x_cpy)
        x = self.conv1(x)
        x = self.bn1(x, training=self.train)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return tf.add(x, x_cpy)


class Skip(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(Skip, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t

        self.pool = layers.MaxPool2D(pool_size=2, strides=2)
        self.conv = layers.Conv2D(filters=self.out_channel, strides=1, kernel_size=1)
        self.bn = layers.BatchNormalization()

    def call(self, x, **kwargs):
        if self.in_channel == self.out_channel and self.stride == 1:
            return x
        if self.in_channel == self.out_channel and self.stride == 2:
            return self.pool(x)
        if self.in_channel != self.out_channel and self.stride == 1:
            return self.bn(self.conv(x))


class Sampled_nn0(keras.Model):
    def __init__(self, t, bs):
        super(Sampled_nn0, self).__init__()
        self.train = t
        self.batch_size = bs

        # 从文件中读取采样结果
        self.pm = [[7, 4, 2, 1, 6, 6, 8, 8, 5, 3, 6, 3, 3, 5, 7, 8, 8],
                   [7, 4, 2, 1, 0, 8, 3, 0, 2, 4, 1, 1, 0, 7, 7, 8, 3],
                   [7, 4, 8, 2, 0, 7, 3, 0, 0, 4, 0, 0, 4, 8, 8, 8, 8],
                   [7, 4, 0, 2, 0, 8, 3, 1, 0, 5, 0, 2, 2, 8, 3, 8, 8],
                   [8, 4, 6, 2, 0, 8, 3, 0, 3, 4, 3, 0, 1, 8, 7, 8, 8],
                   [8, 4, 0, 2, 0, 8, 3, 1, 3, 4, 0, 0, 4, 7, 7, 8, 3],
                   [8, 4, 4, 2, 0, 8, 3, 0, 5, 3, 0, 2, 0, 6, 8, 8, 8]]

        self.result = self.pm[0]
        self.blk_order_list = ['Block_k3_e1', 'Block_k3_e1_g2', 'Block_k3_e3', 'Block_k3_e6',  # 依次对应order中的0-8的序号
                               'Block_k5_e1', 'Block_k5_e1_g2', 'Block_k5_e3', 'Block_k5_e6', 'Block_skip']
        self.in_channel_list = [16, 16, 16, 24, 24, 24, 32, 32, 32, 64, 64, 64,
                                112, 112, 112, 184, 184]
        self.out_channel_list = [16, 16, 24, 24, 24, 32, 32, 32, 64, 64, 64, 112,
                                 112, 112, 184, 184, 352]
        self.stride_list = [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        self.conv1 = layers.Conv2D(filters=self.in_channel_list[0], kernel_size=3,
                                   strides=2, padding='same')
        self.backbone = keras.models.Sequential()

        for i in range(len(self.result)):
            if self.result[i] == 0:
                self.backbone.add(K3_e1(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 1:
                self.backbone.add(K3_e1_g2(i=self.in_channel_list[i],
                                           o=self.out_channel_list[i],
                                           stride=self.stride_list[i], t=self.train))
            if self.result[i] == 2:
                self.backbone.add(K3_e3(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 3:
                self.backbone.add(K3_e6(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 4:
                self.backbone.add(K5_e1(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 5:
                self.backbone.add(K5_e1_g2(i=self.in_channel_list[i],
                                           o=self.out_channel_list[i],
                                           stride=self.stride_list[i], t=self.train))
            if self.result[i] == 6:
                self.backbone.add(K3_e3(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 7:
                self.backbone.add(K3_e6(i=self.in_channel_list[i],
                                        o=self.out_channel_list[i],
                                        stride=self.stride_list[i], t=self.train))
            if self.result[i] == 0:
                self.backbone.add(Skip(i=self.in_channel_list[i],
                                       o=self.out_channel_list[i],
                                       stride=self.stride_list[i], t=self.train))

        self.conv2 = layers.Conv2D(filters=1504, kernel_size=1)
        self.avgpool = layers.AveragePooling2D(pool_size=7)
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(512)
        self.drop = layers.Dropout(0.2)
        self.act1 = layers.ReLU()
        self.fc2 = layers.Dense(17, activation='sigmoid')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act1(x)

        return self.fc2(x)


class Sampled_nn1(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn1, self).__init__(t=t, bs=bs)
        self.result = self.pm[1]


class Sampled_nn2(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn2, self).__init__(t=t, bs=bs)
        self.result = self.pm[2]


class Sampled_nn3(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn3, self).__init__(t=t, bs=bs)
        self.result = self.pm[3]


class Sampled_nn4(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn4, self).__init__(t=t, bs=bs)
        self.result = self.pm[4]


class Sampled_nn5(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn5, self).__init__(t=t, bs=bs)
        self.result = self.pm[5]


class Sampled_nn6(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn6, self).__init__(t=t, bs=bs)
        self.result = self.pm[6]


class Train_data_generator(tf.python.keras.utils.Sequence):
    # 最终版本：使用keras基于Sequence的生成器来加载数据
    def __init__(self, total_train_imgs,
                 train_label_path,
                 batch_size):
        """
        total_train_imgs需要给出图像的完整路径
        其内容为一个列表，列表中的每个元素是一个绝对路径（通过读取次路径获取训练样本）
        """
        self.total_train_imgs = total_train_imgs
        self.train_label_path = train_label_path
        self.batch_size = batch_size
        self.train_length = len(self.total_train_imgs)
        self.on_epoch_end()
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return int(np.ceil(self.train_length / self.batch_size))

    def __getitem__(self, index):
        batch_x0 = self.total_train_imgs[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for item in batch_x0:
            one_img = np.array(Image.open(item)) / 255
            batch_x.append(one_img)

            img_name = item.split('/')[-1]
            f = open(self.train_label_path)
            train_reader = csv.reader(f)
            for row in train_reader:
                if row[0] == img_name.replace('.jpg', '0_2.png'):
                    batch_y.append(np.array(row[3:20], dtype=np.int16))
                    break
            f.close()

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    def one_epoch_end(self):
        pass


class Test_data_generator(tf.python.keras.utils.Sequence):
    def __init__(self, test_imgs, test_label_path, batch_size):
        self.test_imgs = test_imgs
        self.test_label_path = test_label_path
        self.batch_size = batch_size
        self.test_length = len(self.test_imgs)
        self.on_epoch_end()
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return int(np.ceil(self.test_length / self.batch_size))

    def __getitem__(self, index):
        batch_x0 = self.test_imgs[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for item in batch_x0:
            one_img = np.array(Image.open(item)) / 255
            batch_x.append(one_img)

            img_name = item.split('/')[-1]
            f = open(self.test_label_path)
            test_reader = csv.reader(f)
            for row in test_reader:
                if row[0] == img_name.replace('.jpg', '0_2.png'):
                    batch_y.append(np.array(row[3:20], dtype=np.int16))
                    break
            f.close()

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    def one_epoch_end(self):
        pass


def train(n, epoches=2, batch_size=64, init_lr=4e-3):
    """

    :param n: train the nth subnet
    :param epoches: epoches
    :param batch_size: batch size
    :param init_lr: initial learning rate
    :return: None, automatically save the model as 'M_Sampled_nnX.h5', where X mean certain subnet.
    """

    # 数据集模型路径，在不同的环境下一定要改变。
    train_path1 = './train_set/'
    train_label_path = 'SewerML_Train.csv'
    test_path1 = './test_set/'
    test_label_path = 'SewerML_Val.csv'
    model_path = './model_save/M_Sampled_nn' + str(n)

    """
    这里加载数据集的方法经过了多次尝试。
    首先，使用tensorflow的dataset不当，导致训练数据全部进入内存。
    其次，计划使用tensorflow提供的from_tensor_slice没有成功。
    最后决定采用keras的fit_generator和Sequence进行动态加载数据集。
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print('GPU available:', tf.test.is_gpu_available())

    # 用tensorflow提供的keras训练
    if n == 0:
        model = Sampled_nn0(t=True, bs=batch_size)
    elif n == 1:
        model = Sampled_nn1(t=True, bs=batch_size)
    elif n == 2:
        model = Sampled_nn2(t=True, bs=batch_size)
    else:
        model = Sampled_nn3(t=True, bs=batch_size)

    if os.path.exists(model_path):
        model.load_weights(model_path)

    model.compile(optimizer=SGD(init_lr),
                  loss_weights=[[0.2177, 0.0541, 0.6137, 0.5227, 0.0351, 1.5908, 0.4407, 0.4195, 0.1333, 0.1500, 1.9912, 0.1848,
                                 0.4212, 1.4788, 2.1569, 1.8734, 0.0645]],
                  loss=BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # 使用datatset是用此fit方法训练
    # 更改后要相应调整训练方式
    """model.fit(train_dataset, epochs=epoches,
                      validation_data=test_dataset, validation_freq=2)"""

    print('start loading dataset...')
    x_train = []
    x_test = []

    names_in_one_folder = os.listdir(train_path1)
    for item in names_in_one_folder:
        x_train.append(train_path1 + item)

    test_names = os.listdir(test_path1)
    for item1 in test_names:
        x_test.append(test_path1 + item1)
    print('dataset loaded.')

    train_generator = Train_data_generator(total_train_imgs=x_train,
                                           train_label_path=train_label_path,
                                           batch_size=batch_size)
    test_generator = Test_data_generator(test_imgs=x_test,
                                         test_label_path=test_label_path,
                                         batch_size=batch_size)

    print('start training!')
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=int(len(train_generator)),
                        epochs=epoches,
                        validation_data=test_generator,
                        validation_freq=1,
			workers=16,
			use_multiprocessing=True)

    model.save(model_path, save_format='tf')

    model.summary()

def customed_f2ciw(y_true, y_pred):
    y_true01 = K.round(K.clip(y_true, 0, 1))
    y_pred01 = K.round(K.clip(y_pred, 0, 1))

    print(type(y_true01))
    y_true1 = np.array(y_true01)
    y_pred1 = np.array(y_pred01)
    print(typr(y_true1))
    """ytrue_holder = tf.placeholder(dtype=tf.float32, shape=[1, 17])
    ypred_holder = tf.placeholder(dtype=tf.float32, shape=[1, 17])

    ytrue_data = sess.run(y_true01)
    y_true1 = sess.run(feed_dict={ytrue_holder:ytrue_data})
    ypred_data = sess.run(y_pred01)
    y_pred1 = sess.run(feed_dict={ypred_holder: ypred_data})"""

    labels_for_metrics = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    p, r, f2score, s = skmetric.precision_recall_fscore_support(y_true1, y_pred1,
                                                                beta=2.0, labels=labels_for_metrics,
                                                                average=None, zero_division='warn')

    ciw = np.array([1.0000, 0.5518, 0.2896, 0.1622, 0.6419, 0.1847, 0.3559, 0.3131, 0.0811,
                    0.2275, 0.2477, 0.0901, 0.4167, 0.4167, 0.9009, 0.3829, 0.4396])  # from Sewer-ML

    f2ciw = np.dot(f2score, ciw) / np.sum(ciw)
    f2ciw = tf.convert_to_tensor(f2ciw)
    print(f2ciw.dtype)

    del ciw
    return f2ciw


def eval_model(n, batch_size=64):
    """

    :param batch_size: batch_size
    :param n: 评估第n个子网
    :return: 评估结果
    """

    test_path1 = './test_set/'
    test_label_path = 'SewerML_Val.csv'
    model_path = './model_save/M_Sampled_nn' + str(n)

    print('GPU available:', tf.test.is_gpu_available())

    if os.path.exists(model_path):
        model_eval = keras.models.load_model(model_path)
    else:
        raise BlockingIOError('!!!!!!!!!!!!!!Model cannot be loaded!!!!!!!!!!!!!!!!!!!!!!!')


    model_eval.compile(optimizer=SGD(4e-3),
                  loss_weights=[[0.2177, 0.0541, 0.6137, 0.5227, 0.0351, 1.5908, 0.4407, 0.4195, 0.1333, 0.1500,
                                 1.9912, 0.1848, 0.4212, 1.4788, 2.1569, 1.8734, 0.0645]],
                  loss=BinaryCrossentropy(from_logits=False),
                  metrics=[metrics.Precision(), metrics.Recall()])

    x_test = []
    test_names = os.listdir(test_path1)
    for item1 in test_names:
        x_test.append(test_path1 + item1)

    test_generator = Test_data_generator(test_imgs=x_test,
                                         test_label_path=test_label_path,
                                         batch_size=batch_size)

    print('start evaluating')
    model_eval.evaluate(test_generator,
                        workers=16,
                        use_multiprocessing=True)


if __name__ == '__main__':
    train(1)
    eval_model(1)
    train(2)
    eval_model(2)
    train(3)
    eval_model(3)

