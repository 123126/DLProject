"""
演示真实运用场景
"""

import torch

#定义变量（正向传播）
x = torch.ones(2,3,requires_grad=True)  #输入变量
print(f'x:{x}')
y = torch.zeros(2,3,requires_grad=True) #真实值变量
print(f'y:{y}')

w = torch.randn(3,2,requires_grad=True) #随机权重变量
print(f'w:{w}')

b = torch.randn(2,requires_grad=True)
print(f'b:{b}')

#求出z值
z = x @ w + b
print(f'z:{z}')

#反向传播
criterion = torch.nn.MSELoss()
loss = criterion(z,y)
loss.sum().backward()
print(f'x.grad:{x.grad}')
