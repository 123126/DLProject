"""
损失函数介绍：
    分类问题:
        多分类交叉熵损失：CrossEntropyLoss
        二分类交叉熵损失：BCELoss
    回归问题：
        MAE：平均绝对误差     弊端：弊端是在零点不平滑，会错过最小值，因此引出Smooth L1
        MSE：均方误差        弊端：如果差值过大，可能存在梯度爆炸的情况.
        Smooth L1：结合以上两个特点优化升级，在[-1，1]是L2(MSE)，其它段时L1.这样即解决了L1不平滑的问题(0点不可导，可能错过最小值)
                                        又解决了L2(MSE)的 梯度爆炸的问题.


多分类交叉熵损失:CrossEntropyLoss
设计思路:
    Loss = - ∑ ylog(S(f(x)))
简单记忆:
    x   :     样本
    f(x):    加权求和
    S(f(x)): 处理后的概率
    y   :    样本x属于某一个类别的 真实概率

大白话解释:
    损失函数结果= 确类别概率的对数的最小化...
细节:
    CrossEntropyLoss = Softmax() + 损失计算，后续如果用这个损失函数，则:输出层就不用额外调用 softmax()激活函数了.


分类任务的损失函数:BCELoss
公式:
    Loss=-ylog(预测值)-(1-y)Log(1-预测值)
细节:
    因为公式中没有包含Sigmoid激活函数，所以使用BCELoss的时候，还需要手动指定 Sigmoid.



"""
import torch
import torch.nn as nn

def cross_entropy_loss():
    y_true = torch.tensor([1, 2], dtype=torch.long)  #等价于[[0,1,0], [0,0,1]],dtype=torch.float
    y_pred = torch.tensor([[0.1, 0.9, 0.8], [0.1, 0.2, 0.7]], dtype=torch.float)

    citerion = nn.CrossEntropyLoss()
    loss = citerion(y_pred, y_true)

    print(f'loss:{loss}')

def bce_loss():
    y_true = torch.tensor([0, 1, 0],dtype=torch.float)
    y_pred = torch.tensor([0.6985, 0.1235, 0.5641], dtype=torch.float)#二分类任务概率相加!= 1
    citerion = nn.BCELoss()
    loss = citerion(y_pred, y_true)

    print(f'loss:{loss}')

def mae_loss():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], dtype=torch.float , requires_grad=True)  # 二分类任务概率相加!= 1
    citerion = nn.L1Loss()
    loss = citerion(y_pred, y_true)

    print(f'loss:{loss}')

def mse_loss():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], dtype=torch.float , requires_grad=True)  # 二分类任务概率相加!= 1
    citerion = nn.MSELoss()
    loss = citerion(y_pred, y_true)

    print(f'loss:{loss}')

def smooth_l1_loss():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], dtype=torch.float , requires_grad=True)  # 二分类任务概率相加!= 1
    citerion = nn.SmoothL1Loss()
    loss = citerion(y_pred, y_true)

    print(f'loss:{loss}')

if __name__ == '__main__':
    # cross_entropy_loss()
    # bce_loss()
    # mae_loss()
    # mse_loss()
    smooth_l1_loss()