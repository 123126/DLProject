import torch
import torch.nn as nn
from torch.nn import init
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary
BATCH_SIZE = 8


# 1.准备数据集
def create_dataset():
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=False)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=False)

    return train_dataset, test_dataset

# 2.搭建神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1 = nn.Linear(16 * 6 * 6, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # 全连接层只能处理二维数据，所以要把数据拉平 (8, 16, 6, 6) -> (8, 576)
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape:{x.shape}')
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output(x)

        return x




# 3.模型训练
def train(train_dataset, epochs=10):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss_total, batch_num = 0.0, 0
        start = time.time()
        for x, y_true in dataloader:
            model.train()
            y_pred = model(x)
            loss  = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            loss_total += loss
            batch_num += 1
        print(f'epoch:{epoch},loss:{loss_total/batch_num:.4f},time:{time.time() - start:.2f}s')
        torch.save(model.state_dict(), './model/cnn_test.pth')




# 4.模型测试
def evluate(test_dataset):
    model = ImageModel()
    model.load_state_dict(torch.load('./model/cnn_test.pth'))
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct_total, total_num = 0, 0
    for x, y_true in dataloader:
        model.eval()
        y_pred = model(x)
        correct_total += (y_pred.argmax(dim=1) == y_true).sum()
        total_num += len(y_true)
    print(f'准确率：{correct_total/total_num:.4f}')






if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    # print(f'train_dataset.shape:{train_dataset.data.shape}')# (50000, 32, 32, 3)
    # print(f'test_dataset.data.shape:{test_dataset.data.shape}')# (10000, 32, 32, 3)
    # #{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print('train_dataset.classes:', train_dataset.class_to_idx)

    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()

    # model = ImageModel()
    # summary(model, (3, 32, 32), device='cpu')

    # train(train_dataset)
    evluate(test_dataset)


