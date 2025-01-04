import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random

# 人脸情绪识别数据集的标签
data_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        data_dirs_list = glob.glob(os.path.join(data_path, '*'))
        for data_dir_path in data_dirs_list:
            self.data.append(glob.glob(os.path.join(data_dir_path, '*.jpg')))

        self.data = np.concatenate(self.data, axis=0)

        random.shuffle(self.data)

        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path)
        img = np.array(img)
        img_tensor = torch.tensor(img, dtype=torch.float32).view(1, img.shape[0], img.shape[1])
        img_label = torch.tensor(data_labels.index(os.path.normpath(img_path).split(os.sep)[-2]), dtype=torch.long)
        return img_tensor, img_label
