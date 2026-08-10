"""
演示参数初始化的7种方式

参数初始化的目的:
    1.防止梯度消失或者梯度爆炸.
    2.提高收敛速度.
    3.打破对称性.

参数初始化的方式:
    无法打破对称性的:
        全0，全1，固定值
    可以打破对称性的:
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

总结:
    1.记忆kaiming初始化，xavier初始化，全0初始化.
    2.关于初始化的选择上:
        激活函数ReLU及其系列:优先用kaiming
        激活函数非ReLU:优先用 xavier
        如果是浅层网络:可以考虑使用 随机初始化
"""

import torch.nn as nn

#1.均匀分布随机初始化
def demo1():
    linear = nn.Linear(5, 3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#2.固定初始化
def demo2():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 3)
    nn.init.constant_(linear.bias, 3)
    print(linear.weight.data)
    print(linear.bias.data)

#3.全0初始化
def demo3():
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#4.全1初始化
def demo4():
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)


#5.正态分布随机初始化
def demo5():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight)
    print(linear.weight.data)

#6.kaiming初始化
def demo6():
    linear = nn.Linear(5, 3)
    #kaiming正态分布初始化
    # nn.init.kaiming_normal_(linear.weight)
    # print(linear.weight.data)
    #kaiming均匀分布初始化
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data)

#xavier初始化
def demo7():
    linear = nn.Linear(5, 3)
    #xavier正态分布初始化
    # nn.init.xavier_normal_(linear.weight)
    # print(linear.weight.data)
    #xavier均匀分布初始化
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)


if __name__ == '__main__':
    demo7()

