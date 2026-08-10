"""
reshape()       在不改变张量的内容下，对其形状做改变
unsqueez()      在指定地方增加维度，等价升维
squeez()        删除所有为1的维度，等价降维
transpose()     一次只能对换两个维度
permute()       一次能对换多个维度
view()          只能修改连续张量的形状(连续张量指内存张量顺序与现实张量顺序相同)
contiguous()    把不连续张量修改成连续张量
is_contiguous() 判断张量是否连续
"""
import torch
from torch._prims_common import is_contiguous

torch.manual_seed(24)

#1.reshape()
def demo1():
    t1 = torch.randint(0, 10, size = (2,3))
    print(f"t: {t1}, row: {t1.shape[0]}, col: {t1.shape[-1]}")

    #通过reshape改变形状
    t2 = t1.reshape(3,2)
    print(f"t2: {t2}, row: {t2.shape[0]}, col: {t2.shape[-1]}")

#2.unsqueez()和squeez()
def demo2():
    #unsqueez()函数
    t1 = torch.randint(0, 10, size=(2, 3))
    print(f"t: {t1}, shape: {t1.shape}")
    t2 = t1.unsqueeze(0)
    print(f"t2: {t2}, shape: {t2.shape}")
    t3 = t1.unsqueeze(1)
    print(f"t3: {t3}, shape: {t3.shape}")

    #squeez()函数
    t4 = torch.randint(0, 10, size=(2, 1, 3, 1, 1))
    print(f"t4: {t4}, shape: {t4.shape}")
    t5 = t4.squeeze()
    print(f"t5: {t5}, shape: {t5.shape}")



#3.transpose()和permute()
def demo3():
    t1 = torch.randint(0, 10, size=(2, 3, 4))
    print(f"t: {t1}, shape: {t1.shape}")
    t2 = t1.transpose(1, 2) #(2, 3, 4) -> (2, 4, 3)
    print(f"t2: {t2}, shape: {t2.shape}")

    t3 = t1.permute(0, 2, 1)
    print(f"t3: {t3}, shape: {t3.shape}")

#4.view()和contiguous()和is_contiguous()
def demo4():
    t1 = torch.randint(0, 10, size=(2, 3))
    print(is_contiguous(t1))
    print(f"t: {t1}, shape: {t1.shape}")
    t2 = t1.view(3, 2)
    print(f"t2: {t2}, shape: {t2.shape}")


    t3 = t1.transpose(1, 0)
    print(f"t3: {t3}, shape: {t3.shape}")

    t4 = t3.contiguous().view(2, 3)
    print(f"t4: {t4}, shape: {t4.shape}")

if __name__ == '__main__':
    demo4()