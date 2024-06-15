from tensorflow.python import keras
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.models import load_model, save_model

import os
import numpy as np
from PIL import Image
import csv


class K3_e1(layers.Layer):
    def __init__(self, i, o, stride, t):
        super(K3_e1, self).__init__()
        self.in_channel = i
        self.out_channel = o
        self.stride = stride
        self.train = t
        self.expansion = 1

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid', groups=2)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=2)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=3,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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

        self.conv1 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=1,
                                   strides=self.stride, padding='valid', groups=2)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=2)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
                                   strides=self.stride, padding='valid', groups=1)
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU(negative_slope=1e-4)            # leaky relu
        self.conv2 = layers.Conv2D(filters=self.in_channel * self.expansion, kernel_size=5,
                                   strides=1, padding='same', groups=self.expansion * self.in_channel)
        self.conv3 = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                   strides=1, padding='same', groups=1)
        self.dimkeep_conv = layers.Conv2D(filters=self.out_channel, kernel_size=1,
                                          strides=self.stride, groups=1)

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
        if self.in_channel==self.out_channel and self.stride==1:
            return x
        if self.in_channel==self.out_channel and self.stride==2:
            return self.pool(x)
        if self.in_channel!=self.out_channel and self.stride==1:
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
        self.result = self.pm[1]

class Sampled_nn3(Sampled_nn0):
    def __init__(self, t, bs):
        super(Sampled_nn3, self).__init__(t=t, bs=bs)
        self.result = self.pm[1]



tf.flags.DEFINE_string('data_url', '../dataset/', 'dataset')
tf.flags.DEFINE_string('train_url', '../output/', 'model_save')

F = tf.flags.FLAGS

def train(n, epoches=20, batch_size=64, init_lr=0.01):
    """

    :param n: train the nth subnet
    :param epoches: epoches
    :param batch_size: batch size
    :param init_lr: initial learning rate
    :return: None, automatically save the model as 'M_Sampled_nnX', where X mean certain subnet.
    """

    # 先加载数据集，必须采用dataset否则数据集太大无法送入RAM
    print('start loading dataset...')
    train_imgs_path1 = F.data_url + 'train_set'
    test_imgs_path = F.data_url + 'test_set/'
    train_label_path = F.data_url + 'SewerML_Train.csv'
    test_label_path = F.data_url + 'SewerML_Val.csv'
    model_path = F.train_url + 'M_Sampled_nn' + str(n) + '.h5'

    x_train = []    ;   y_train = []    ;   x_test = []     ;   y_test = []

    f_train = open(train_label_path)
    f_test = open(test_label_path)

    train_reader = csv.reader(f_train)
    for i in range(11):
        train_img_path = train_imgs_path1 + str(i) + '/'
        train_img_names = os.listdir(train_imgs_path)
        for item in train_img_names:
            img = np.array(Image.open(train_imgs_path+item), dtype=np.float16) / 255
            x_train.append(img)
            for row in train_reader:
                if row[0] == item.replace('.jpg', '0_2.png'):
                    y_train.append(np.array(row[3:20], dtype=np.int16))
                    break

    test_reader = csv.reader(f_test)
    test_img_names = os.listdir(test_imgs_path)
    for item in test_img_names:
        img = np.array(Image.open(test_imgs_path+item), dtype=np.float16) / 255
        x_test.append(img)
        for row in test_reader:
            if row[0] == item.replace('.jpg', '0_2.png'):
                y_test.append(np.array(row[3:20], dtype=np.int16))
                break

    f_train.close()
    f_test.close()
    print('dataset loaded.')        # 得到xtrain ytrain xtest ytest

    # 用tensorflow的dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1100000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # 用tensorflow提供的keras训练
    if not os.path.exists(model_path):
        if n == 0:
            model = Sampled_nn0(t=True, bs=batch_size)
        elif n == 1:
            model = Sampled_nn1(t=True, bs=batch_size)
        elif n == 2:
            model = Sampled_nn2(t=True, bs=batch_size)
        elif n == 3:
            model = Sampled_nn3(t=True, bs=batch_size)
        elif n == 4:
            model = Sampled_nn4(t=True, bs=batch_size)
        elif n == 5:
            model = Sampled_nn5(t=True, bs=batch_size)
        else:
            model = Sampled_nn6(t=True, bs=batch_size)
    else:
        model = load_model(model_path)

    model.compile(optimizer=Adam(learning_rate=ExponentialDecay(
                                                            init_lr, decay_steps=2, decay_rate=0.94)),
                  loss_weights=[21.77, 5.41, 61.37, 52.27, 3.51, 159.08, 44.07, 41.95, 13.33, 15.00, 199.12, 18.48,
                                42.12, 147.88, 215.69, 187.34, 6.45],
                  loss=BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=epoches,
              validation_data=test_dataset, validation_freq=2)

    save_model(model, model_path)

    model.summary()


if __name__ == '__main__':
    train(0, epoches=36)            # 执行训练网络的脚本
