"""
演示detach()的功能，解决自动微分的弊端
自动微分的弊端：
    一个张量一旦设置自动微分，就不能直接转换成numpy的ndarray对象了，需要通过detach()函数解决
"""
import torch
import numpy as np

#定义张量
t1 = torch.tensor([10, 20], requires_grad=True, dtype= torch.float)
print(f't1:{t1}, type:{type(t1)}')

#转化成numpy对象
# n1 = t1.numpy()     #会报错
# print(f'n1:{n1}, type:{type(n1)}')

t2 = t1.detach()  #浅拷贝，共享内存但是requires_grad=Flase ，所以可以转换成numpy对象

n1 = t2.numpy()
print(f'n1:{n1}, type:{type(n1)}')


#同理, t1.detach().numpy()即可
