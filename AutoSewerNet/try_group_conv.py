import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

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
        self.conv1 = layers.Conv2D(filters=self.filters/2, kernel_size=self.kernel_size,
                                   strides=self.strides, padding=self.padding)
        self.conv2 = layers.Conv2D(filters=self.filters/2, kernel_size=self.kernel_size,
                                   strides=self.strides, padding=self.padding)

    def call(self, x, **kwargs):
        channels =x.shape[-1]
        x_split1 = x[:, :, :, :int(channels/2)]
        x_split2 = x[:, :, :, int(channels/2):]

        x_split1 = self.conv1(x_split1)
        x_split2 = self.conv2(x_split2)

        return tf.concat([x_split1, x_split2], axis=3)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv0 = layers.Conv2D(filters=8, kernel_size=3)
        self.conv1 = Grouped_conv2d(filters=8)
        self.flat = layers.Flatten()
        self.fc1 = layers.Dense(10)

    def call(self, x1, **kwargs):
        x1 = self.conv0(x1)
        x1 = self.conv1(x1)
        x1 = self.flat(x1)
        return self.fc1(x1)

x1 = np.empty(shape=(1, 16, 16, 3))
y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
model = Model()
model.compile()
model.fit(x1, y)
model.summary()
