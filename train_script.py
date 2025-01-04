import os
from torch.utils.data import DataLoader
from classifier_models.Convnext import Net
from classifier_models.utils.my_datasets import MyDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


# 训练结果保存路径
train_result_path = './train_result'
os.makedirs(train_result_path, exist_ok=True)


# 数据集路径
train_data_path = './archive/train'
test_data_path = './archive/test'

# 数据集
train_dataset = MyDataset(train_data_path)
test_dataset = MyDataset(test_data_path)

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epochs_num=10, lr=0.001):
    # 模型
    model = Net(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs_num):
        model.train()
        for i, (img, label) in enumerate(train_data):
            img, label = img.to(device), label.to(device)

            optimzer.zero_grad()

            pred_label = model(img)
            loss = criterion(pred_label, label)
            loss.backward()
            optimzer.step()

            if i % 100 == 0:
                label_accurency = (torch.argmax(pred_label, dim=1) == label).sum().item() / label.shape[0]
                single_acc = np.mean([F.softmax(pred_label, dim=1)[j, value].item() for j, value in enumerate(label)])
                print(f'train: epoch: {epoch}, step: {i}, loss: {loss.item()}, accurency: {label_accurency}, single_acc: {single_acc}')

        # 每5个epoch保存一次模型并测试
        if ((epoch != 0) and (epoch % 5 == 0)) or (epoch == epochs_num - 1):
            # save model
            if epoch == epochs_num - 1:
                torch.save(model, os.path.join(train_result_path, f'model_final.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(train_result_path, f'model_{epoch}.pth'))
            # eval
            with torch.no_grad():
                model.eval()
                for i, (img, label) in enumerate(test_data):
                    img, label = img.to(device), label.to(device)

                    pred_label = model(img)

                    if i % 50 == 0:
                        label_accurency = (torch.argmax(pred_label, dim=1) == label).sum().item() / label.shape[0]
                        single_acc = np.mean([F.softmax(pred_label, dim=1)[j, value].item() for j, value in enumerate(label)])
                        print(f'eval: epoch: {epoch}, step: {i}, accurency: {label_accurency}, single_acc: {single_acc}')



if __name__ == '__main__':
    train(epochs_num=10, lr=0.001)
