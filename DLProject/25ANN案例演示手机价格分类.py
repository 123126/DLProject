"""
背景:
    基于手机的20列特征 预测手机的价格区间(4个区间)，可以用机器学习做，也可以用 深度学习做(推荐)

ANN案例的实现步骤:
1.构建数据集.
2.搭建神经网络.
3.模型训练.
4.模型测试.
"""

import torch                                #PyTorch框架，封装了张量的各种操作
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset  #数据集对象.
from torch.utils.data import DataLoader     #数据加载器.
import torch.nn as nn                       #neural network，封装了神经网络的各种操作
import torch.optim as optim                 #优化器
from sklearn.model_selection import train_test_split    # 训练集和测试集的划分
import matplotlib.pyplot as plt             #绘图
import numpy as np                          #数组(矩阵)操作
import pandas as pd                         #数据处理
import time                                 #时间模块
from torchsummary import summary

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# todo 1.定义函数，构建数据集.
def create_dataset():
    data = pd.read_csv('./data/train.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1]  # 切割数据，x是输入特征，y是标签
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 参1：数据集，参2：标签，参3：测试集比例，参4：随机数种子,参5：样本的分布(即参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    #数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 把数据封装成张量数据集
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))

    return train_dataset, test_dataset, x_train.shape[1], len(y_train.unique())  # x_train.shape[1]是输入特征数, len(y_train.unique())是标签类别数


# todo 2.搭建神经网络.
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        # 加权求和+激活函数
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # x = torch.softmax(self.output(x), dim=-1)   #后面使用交叉熵损失函数，所以不需要softmax
        x = self.output(x)

        return x


# todo 3.模型训练.
def train(train_dataset, input_dim, num_classes, epochs=50):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    citerion = nn.CrossEntropyLoss()
    loss_list,epochs_list = [],[]

    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x_train, y_train in train_loader:
            model.train()  # 训练模式
            y_pred = model(x_train)
            loss = citerion(y_pred, y_train)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1


        print(f'epoch:{epoch},loss:{total_loss / batch_num:.4f},time:{time.time() - start:.2f}s')
        epochs_list.append(epoch)
        loss_list.append(total_loss / batch_num)


    # 保存模型  model.state_dict()表示model的参数(权重矩阵)
    torch.save(model.state_dict(), './model/phone_price_model.pth')
    print(f'\n\n模型信息：{model.state_dict()}\n\n')
    plt.plot(epochs_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()


# todo 4.模型测试.
def evaluate(test_dataset, input_dim, num_classes):
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    model = PhonePriceModel(input_dim, num_classes)
    model.load_state_dict(torch.load('./model/phone_price_model.pth'))
    correct = 0
    for x, y in test_loader:
        model.eval()
        y_pred = model(x)
        # print(f'y_pred:{y_pred}') # 得出的是加权求和后，每类别的预测结果
        y_pred = torch.argmax(y_pred, dim=1) #dim=1 按行处理，# 用argmax()获取加权求和最大值的索引
        print(f'y_pred:{y_pred}')
        print(f'y:{y}')
        correct += (y_pred == y).sum()
    print(f'准确率：{correct / len(test_dataset):.4f}')

if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, num_classes = create_dataset()
    print(f'input_dim:{input_dim},num_classes:{num_classes}')
    print('train_dataset:', train_dataset)
    print('test_dataset:', test_dataset)

    model = PhonePriceModel(input_dim, num_classes)
    summary(model, (16, input_dim)) # 参1：模型，参2：(每批数量，输入特征数)

    train(train_dataset, input_dim, num_classes, 50)
    evaluate(test_dataset, input_dim, num_classes)



