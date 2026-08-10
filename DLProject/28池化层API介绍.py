"""
池化不会改变通道数
"""
import torch
import torch.nn as nn

#演示单通道池化
def demo():
    input = torch.tensor([
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
    ])          # [1, 3, 3]
    print(f'input:{input},input.shape:{input.shape}')
    pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = pool(input)# [1, 2, 2]
    print(f'output:{output},output.shape:{output.shape}')

    pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = pool(input)
    print(f'output:{output},output.shape:{output.shape}')



#演示多通道池化
def demo2():
    input = torch.tensor([
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],
        [
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ]

    ])  # [3, 3, 3]
    print(f'input:{input},input.shape:{input.shape}')
    pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = pool(input)  # [3, 2, 2]
    print(f'output:{output},output.shape:{output.shape}')

    pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = pool(input)# [3, 2, 2]
    print(f'output:{output},output.shape:{output.shape}')

if __name__ == '__main__':
    # demo()
    demo2()

