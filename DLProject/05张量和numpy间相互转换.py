import torch
import numpy as np
#1.张量->numpy
def demo1():
    t1 = torch.tensor([1,2,3,4,5,6])
    print(f't1:{t1},type:{type(t1)}')

    # n1 = t1.numpy()         #浅拷贝
    n1 = t1.numpy().copy()  #深拷贝
    print(f'n1:{n1},type:{type(n1)}')

    #修改numpy中的数据,查看是否为深浅拷贝
    n1[0] = 100
    print(f't1:{t1}')
    print(f'n1:{n1}')

#2.numpy->张量
def demo2():
    n1 = np.array([11, 22, 33])
    print(f'n1:{n1},type:{type(n1)}')

    # t1 = torch.from_numpy(n1).type(torch.float)#转换为张量+转换数据类型
    t1 = torch.from_numpy(n1)           #共享内存
    print(f't1:{t1},type:{type(t1)}')

    t2 = torch.tensor(n1)               #不共享内存
    print(f't2:{t2},type:{type(t2)}')

    #查看t1和t2是否共享内存
    n1[0] = 100
    print(f'n1:{n1}')
    print(f't1:{t1}')
    print(f't2:{t2}')

#3.提取标量张量中的数据
def demo3():
    t1 = torch.tensor(100)
    print(f't1:{t1},type:{type(t1)}')

    value = t1.item()
    print(f'value:{value},type:{type(value)}')

if __name__ == '__main__':
    demo3()
