"""
sum(),min(),max(),mean()                     ->   都有dim参数(按行或列做)，0表示列，1表示行
pow(),sqrt(),exp(),log(),log2(),log10()      ->   没有dim参数
"""

import torch

t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float) #取平均数时有可能取到小数

#有dim参数
print(t1.sum(dim=0))        #列求和
print(t1.sum(dim=1))        #行求和
print(t1.sum())        #所有元素求和
print('*'*50)

print(t1.mean(dim=0))        #列求平均值
print(t1.mean(dim=1))        #行求平均值
print(t1.mean())        #所有元素求平均值
print('*'*50)

#没有dim参数
print(t1.pow(2))
print(t1.sqrt())