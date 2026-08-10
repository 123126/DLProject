import torch
import numpy as np
#tensor要给定数据才能创建张量
#而Tensor能根据形状直接创建张量

def demo1():
    t1 = torch.tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-"*30)

    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t1:{t2},type:{type(t2)}")
    print("-"*30)

    data = np.random.randint(0,10,size=(2,3))
    t3= torch.tensor(data,dtype=torch.float)   #浮点类型
    print(f"t1:{t3},type:{type(t3)}")
    print("-"*30)

    #会报错
    # t4 = torch.Tensor(2,3)
    # print(f"t1:{t4},type:{type(t4)}")
def demo2():
    t1 = torch.Tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-"*30)

    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t1:{t2},type:{type(t2)}")
    print("-"*30)

    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.Tensor(data)   #浮点类型
    print(f"t1:{t3},type:{type(t3)}")
    print("-" * 30)

    #不报错
    t4 = torch.Tensor(2,3)
    print(f"t1:{t4},type:{type(t4)}")

def demo3():
    t1 = torch.IntTensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f"t1:{t2},type:{type(t2)}")
    print("-" * 30)

    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)  # 整数类型
    print(f"t1:{t3},type:{type(t3)}")
    print("-" * 30)

def demo4():
    t1 = torch.IntTensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f"t1:{t2},type:{type(t2)}")
    print("-" * 30)

    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)  # 浮点类型
    print(f"t1:{t3},type:{type(t3)}")
    print("-" * 30)

    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)  # 浮点类型 默认float32
    print(f"t1:{t4},type:{type(t4)}")
    print("-" * 30)

if __name__ == '__main__':
    demo2()


