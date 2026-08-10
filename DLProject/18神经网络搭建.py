"""
演示神经网络搭建
"""

import torch
import torch.nn as nn
from torchsummary import summary

#todo 1.搭建神经网络，即：自定义继承 nn.Module
class ModelDemo(nn.Module):
    #在init魔法方法中,完成初始化：父类成员，及神经搭建
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)

        #对隐藏层进行参数初始化,输出层一般用默认初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #todo 1.2. 定义前向传播,计算出加权求和(方法会被自动调用，所以函数名不能自行更改)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))

        #dim = -1表示对最后一维，一条一条样本处理
        x = torch.softmax(self.output(x), dim=-1)
        return x

#todo 2.入门模型训练
def train():
    #创建模型对象
    model = ModelDemo()
    data = torch.randn(5, 3)      #随机生成5行3列作为输入数据
    print(f'data:{data}, data.shape:{data.shape}, data.requires_grad:{data.requires_grad}')

    output = model(data)   #会自动调用forward方法，输出结果为5行2列，并且会自动开启梯度计算
    print(f'output:{output}, output.shape:{output.shape}, output.requires_grad:{output.requires_grad}')

    print("==================查看模型参数==================")
    summary(model, (5, 3))
    print("==================查看模型参数==================")
    for name, param in model.named_parameters():
        print(f'name:{name}, param:{param}, param.shape:{param.shape}\n')


if __name__ == '__main__':
    train()