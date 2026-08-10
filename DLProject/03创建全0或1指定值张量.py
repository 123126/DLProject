import torch

def demo1():
    #创建全1张量
    t1 = torch.ones(2,3)
    print(f't1:{t1},type:{type(t1)}')
    print('-'*30)

    #创建形如t2的全一张量
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    t3 = torch.ones_like(t2)
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    #-------------------------------------------------------
    #创建全0张量
    t1 = torch.zeros(2, 3)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    # 创建形如t2的全0张量
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    t3 = torch.zeros_like(t2)
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    # -------------------------------------------------------
    # 使用full和full_like创建张量
    t1 = torch.full(size=(2, 3),fill_value=255)#创建2行3列255向量
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    # 创建形如t2的全0张量
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    t3 = torch.full_like(t2,fill_value=255)
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)


if __name__ == '__main__':
    demo1()