import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ) # 48*48 -> 24*24

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ) # 24*24 -> 12*12

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ) # 12*12 -> 6*6

        self.pool = nn.MaxPool2d(kernel_size=6)

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    arr = torch.randn(1, 1, 48, 48)
    model = Net()
    output = model(arr)
    print(output)
