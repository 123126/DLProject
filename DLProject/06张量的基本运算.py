"""
涉及到的API：
    add(),sub(),mul(),div(),neg()       ->    加减乘除取反
    add(),sub(),mul(),div(),neg()       ->    功能同上，但可以修改源数据
"""

import torch

t1 = torch.tensor([1, 2, 3])

#1.加法（加减乘除同理）
# t2 = t1.add(2)      #t1所有元素+2，不会修改源数据
# t2 = t1+2           #效果同上

t2 = t1.add_(2)     #t1所有元素+2，修改源数据
t1 += 2             #效果同上

#2.取反
t2 = t1.neg()

print(f"t1:{t1}")
print(f"t2:{t2}")
