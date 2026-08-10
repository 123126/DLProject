"""
演示自动微分模块具体如何求导
w新 = w旧 - 学习率 * 梯度
使用pytorch中自动微分模块更新w和b实现反向传播

注意：只有标量张量才能求导，且大多数底层操作为浮点型
"""

import torch

#1.定义变量
    #参数1：赋值 ， 参数2：是否需要自动求导 ， 参数3：数据类型
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)
    #简单设置一个损失函数，其导数为4w
loss = 2 * w ** 2

#2.开启自动微分模块,将结果保存到w.grad中,sum()将向量计算成标量
loss.sum().backward()

#3.更新w，简单设置学习率0.01
w.data = w.data - 0.01 * w.grad

print(f"w:{w}")