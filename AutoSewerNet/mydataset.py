import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import csv
from PIL import Image
from PIL import ImageFile

class Sampled_train_dataset(Dataset):
    def __init__(self):
        self.meanings = ['RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN',
                         'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK']
        self.train_path = '../../datasets/train_set/'
        self.train_label_path = '../../datasets/SewerML_Train.csv'
        self.train_reader = csv.reader(open(self.train_label_path))
        self.test_path = '.../../datasets/test_set/'
        self.test_label_path = '../../datasets/SewerML_Val.csv'
        self.test_reader = csv.reader(open(self.test_label_path))

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        video_name_l = os.listdir(self.train_path)
        x = np.resize(np.array(Image.open(self.train_path + video_name_l[idx]).convert('RGB'), dtype=np.float32),
                      new_shape=(224, 224, 3)).swapaxes(0, 2)

        y = 1
        while y == 1:
            for rows in self.train_reader:
                if rows[0] == video_name_l[idx].replace('.jpg', '.png'):
                    y = np.array([int(x) for x in rows[3:20]], dtype=np.float32)
                    break
            idx += 1

        return x, y

    def __len__(self):
        return len(os.listdir(self.train_path)) - 1

class Sampled_test_dataset(Dataset):
    def __init__(self):
        self.meanings = ['RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN',
                         'AF', 'BE', 'FO', 'GR', 'PH', 'PB', 'OS', 'OP', 'OK']

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        video_name_l = os.listdir(test_path)
        x = np.resize(np.array(Image.open(test_path + video_name_l[idx]).convert('RGB'), dtype=np.float32),
                      new_shape=(224, 224, 3)).swapaxes(0, 2)
        y = 1
        for rows in self.test_reader:
            if rows[0] == video_name_l[idx].replace('.jpg', '.png'):
                y = np.array([int(x) for x in rows[3:20]], dtype=np.float32)
                break

        return x, y

    def __len__(self):
        return len(os.listdir(self.test_path)) - 1
