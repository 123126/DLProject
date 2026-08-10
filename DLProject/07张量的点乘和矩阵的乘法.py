"""
点乘： t1*t2  /   t1.mul(t2)
乘法:  t1@t2  /   t1.matmul(t2)
"""

import torch


#定义点乘函数
def demo1():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])

    t3 = t1*t2
    print(f"t3:{t3}")

#定义矩阵乘法
def demo2():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])

    t3 = t1 @ t2
    print(f"t3:{t3}")

if __name__ == '__main__':
    demo2()