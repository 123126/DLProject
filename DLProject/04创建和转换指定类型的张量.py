import torch

def demo():
    #直接创建指定类型张量
    t1 = torch.tensor([1, 2, 3, 4, 5, 6],dtype=torch.float)
    print(f't1:{t1}, dtype:{t1.dtype}, type:{type(t1)}')
    print('-'*30)

   #转化张量类型
    t2 = t1.type(torch.int32)
    print(f't2:{t2}, dtype:{t2.dtype}, type:{type(t2)}')
    print('-' * 30)

if __name__ == '__main__':
    demo()