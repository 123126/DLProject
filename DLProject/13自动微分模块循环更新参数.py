"""
演示自动微分模块循环更新参数
求出 y = x ** 2 + 20 的极小值点，并打印出该点的梯度
"""

import torch

#1.定义变量
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)
loss = w ** 2 + 20

#2.循环更新参数
for i in range(500):
    # 2.1 清空梯度（否则默认会累加）第一次循环w.grad为None,所以需要判断
    if i:
        w.grad.zero_()

    #2.2 计算梯度
    loss = w ** 2 + 20
    loss.sum().backward()

    #2.3 更新参数
    w.data = w.data - 0.01 * w.grad
    print(f'第{i}次循环 , w:{w.data}, grad:{w.grad}, loss:{loss}')



print(f'最终结果:w:{w.data}, grad:{w.grad}, loss:{loss}')