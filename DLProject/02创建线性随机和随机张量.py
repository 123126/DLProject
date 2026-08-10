import torch

def demo1():
    #创建线性张量
    t1 = torch.arange(0, 10, 4)
    print(f't1:{t1},type:{type(t1)}')
    print('-'*30)

    t2 = torch.linspace(0, 10, 4)
    print(f't2:{t2},type:{type(t2)}')

def demo2():
    # todo 1.采用随机种子
    # torch.initial_seed()#默认使用时间戳作为随机种子
    torch.manual_seed(3)#设置随机种子

    # todo 2.创建随机张量
    t1 = torch.rand(size=(2,3))         #均匀分布的随机向量
    print(f't1:{t1},type:{type(t1)}')
    print('-'*30)

    t2 = torch.randn(size=(2, 3))       #符合正态分布的随机向量
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

    t3 = torch.randint(1, 10, size=(3, 5))       #随机创建整数向量
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)




if __name__ == '__main__':
    demo2()
