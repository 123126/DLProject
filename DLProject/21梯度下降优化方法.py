"""
梯度下降相关介绍:
    概述:
        梯度下降是结合 本次损失函数的导数(作为梯度)基于学习率来更新权重的.
    公式:
        W新=W旧-学习率*(本次的)梯度
    存在的问题(梯度下降算法优化目的):
        1.遇到平缓区域，梯度下降(权重更新)可能会慢
        2.可能会遇到 鞍点(梯度为0)
        3.可能会遇到 局部最小值 导致梯度始终为0不改变
    解决思路:
        从上述的 学习率或者 梯度入手，进行优化，于是有了:动量法Momentum，自适应学习率AdaGrad，RMSProp，综合衡量:Adam

    动量法Momentum（改变梯度）:
        动量法公式:
            St = β * St-1 + (1 - β) * Gt
        解释:
            St: 本次的指数移动加权平均结果.调节权重系数，越大，数据越平缓，历史指数移动加权平均 比重越大，本次梯度权重越小.
            St-1:历史的指数移动加权平均结果.
            Gt:本次计算出的梯度(不考虑历史梯度).
            β:  权重系数
        加入动量法后的 梯度更新公式:
            W新=W旧-学习率*St

    自适应学习率:AdaGrad
        公式:
            累计平方梯度:
                St = St-1 + Gt * Gt
            解释:
                St:     累计平方梯度
                St-1:   历史累计平方梯度.
                Gt:     本次的梯度.
            学习率:
                学习率=学习率/(sqrt(St)+小常数)
            解释:
                小常数:1e-10(十的负十次方)，目的:防止分母变为0
            梯度下降公式:
                W新=W旧- 调整后的学习率*Gt
        缺点:
            可能会导致学习率过早，过量的降低，导致模型后期学习率太小，较难找到最优解。

    自适应学习率:RMSProp  可以看做是 对AdaGrad做的优化，加入 调和权重系数.
        公式:
            指数加权平均累计历史平方梯度:
                St =β* St-1+ (1 -β)* Gt * Gt
            解释:

                St:累计平方梯度
                St-1:历史累计平方梯度。
                Gt:本次的梯度.
                β:调和权重系数.
            学习率:
                学习率=学习率/(sqrt(St)+小常数)
            解释:
                小常数:1e-10，目的:防止分母变为0
            梯度下降公式:
                W新=W旧- 调整后的学习率*Gt
        优点:
            RMSProp通过引入 衰减系数，控制历史梯度对 历史梯度信息获取的多少.

    自适应矩估计:Adam(Adaptive Moment Estimation)
        思路:
            即优化学习率，又优化梯度.
        公式
            一阶矩:算均值.
                Mt = β1 * Mt-1+ (1 - β1)* Gt            充当:梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt     充当:学习率
            二阶矩:梯度的方差.
                Mt^=Mt / (1 - β1^t)
                St^=St / (1 - β2^t)
            权重更新公式:
                W新=W旧-学习率/(sqrt(St^)+小常数)* Mt^
        大白话翻译:
            Adam = RMSProp + Momentum
"""
import torch
import torch.nn as nn


# 1.演示梯度下降优化方法：动量法
def momentum_optimizer():
    w = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
    # 第一次梯度计算:
    citerion = (w ** 2) / 2

    """
    # 参1：待优化的参数，参2：学习率，参3：动量系数
    # torch.optim.SGD([w], lr=0.01, momentum=0.9, dampening=0.9)添加了dampening，就变成St = β * St-1 + (1 - β) * Gt
    # 在实际的工程中，SGD会省略（1+β）项，即St = β * St-1 + Gt
    """
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    # 梯度清零+梯度计算+反向传播
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()

    print(f'w:{w},w.grad:{w.grad}')

    # 第二次梯度计算:
    citerion = (w ** 2) / 2
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')


# 2.演示自适应学习率AdaGrad
def adagrad_optimizer():
    w = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
    # 第一次梯度计算:
    citerion = (w ** 2) / 2
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    # 梯度清零+梯度计算+反向传播
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()

    print(f'w:{w},w.grad:{w.grad}')

    # 第二次梯度计算:
    citerion = (w ** 2) / 2
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 3.演示自适应学习率RMSProp
def rmsprop_optimizer():
    w = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
    # 第一次梯度计算:
    citerion = (w ** 2) / 2
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.99)# alpha相当于动量法的β
    # 梯度清零+梯度计算+反向传播
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()

    print(f'w:{w},w.grad:{w.grad}')

    # 第二次梯度计算:
    citerion = (w ** 2) / 2
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 4.演示自适应矩估计Adam
def adam_optimizer():
    w = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
    # 第一次梯度计算:
    citerion = (w ** 2) / 2
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.99))  # alpha相当于动量法的β
    # 梯度清零+梯度计算+反向传播
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()

    print(f'w:{w},w.grad:{w.grad}')

    # 第二次梯度计算:
    citerion = (w ** 2) / 2
    optimizer.zero_grad()
    citerion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

if __name__ == '__main__':
    # momentum_optimizer()
    # adagrad_optimizer()
    # rmsprop_optimizer()
    adam_optimizer()
