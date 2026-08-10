"""
正则化方式：
	L1正则化:权重可以变为0，相当于:降维.
	L2正则化:权重可以无限接近0
	Drop0ut:随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影响较大(防止过拟合)
	BN(批量归一化):

"""

import torch
import torch.nn as nn
from torch.nn.modules import dropout


#演示DropOut随机失活
def DropOut_demo():
    input_data = torch.randint(0, 10, size =(1,4)).float()

    #创建隐藏层+加权求和+激活函数
    linear1 = nn.Linear(4,5)
    output = torch.relu(linear1(input_data))
    print(f'linear1:{linear1(input_data).data}')
    print(f'output:{output.data}')

    #Dropout正则化 参1：失活概率
    dropout = nn.Dropout(p=0.5)
    d1 = dropout(output)
    print(f'dropout:{d1}')




if __name__ == '__main__':
    DropOut_demo()