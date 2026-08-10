"""
torch.cat()     不改变维度数，拼接张量，除了拼接的那个维度外，其他维度必须保持一致
torch.stack()   会改变维度数，拼接张量，所有维度都必须保持一致
"""

import torch
torch.manual_seed(24)

t1 = torch.randint(0, 10, size=(2, 3))
print(f"t1:{t1}, shape: {t1.shape}")

t2 = torch.randint(0, 10, size=(2, 3))
print(f"t2:{t2}, shape: {t2.shape}")

#1.演示torch.cat()
# (2,3)+(2,3)=(4,3)在第零维进行拼接
t3 = torch.cat([t1, t2], dim=0)
print(f"t3:{t3}, shape: {t3.shape}")

t4 = torch.cat([t1, t2], dim=1)
print(f"t4:{t4}, shape: {t4.shape}")
print("-" * 30)

#2.演示torch.stack()
t5 = torch.stack([t1, t2], dim=0)  # (2,3)+(2,3)=(2,2,3) 将2个张量堆叠在第零维
print(f"t5:{t5}, shape: {t5.shape}")
t6 = torch.stack([t1, t2], dim=1)   # (2,3)+(2,3)=(2,2,3) 将2个张量堆叠在第1维
print(f"t6:{t6}, shape: {t6.shape}")
t7 = torch.stack([t1, t2], dim=2)  # (2,3)+(2,3)=(2,3,2) 将2个张量堆叠在第2维
print(f"t7:{t7}, shape: {t7.shape}")
