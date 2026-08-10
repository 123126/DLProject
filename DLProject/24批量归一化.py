"""
批量归一化思路： 先对数据做归一化处理（会丢失一些信息），再对数据做 缩放和平移 （再找回一些信息）
批量归一化在cv中常用

BatchNormld:主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为(N，num_features)的张量作为输入。
BatchNorm2d:主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为(N，C,H，W)的张量作为输入。
BatchNorm3d:主要用于三维卷积神经网络(3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为(N，C，D，H，W)的张量作为输入。
"""
import torch
import torch.nn as nn

#演示BatchNorm2d
def batch_norm_demo():
    input_2d = torch.randn(size=(1, 2, 3, 4))
    """
    # num_features: 图片通道数
    # momentum： 动量系数
    # eps： 防止除零误差的常数
    # affine： 是否使用可学习的缩放和平移参数
    """
    batch_norm = nn.BatchNorm2d(num_features=2, momentum=0.5, eps=1e-05, affine=True)
    output = batch_norm(input_2d)
    print(f'output:{output}, output.shape:{output.shape}')

#演示BatchNorm1d
def batch_norm_1d_demo():
    input_1d = torch.randn(size=(2, 2))
    print(f'input_1d:{input_1d}')
    batch_norm = nn.BatchNorm1d(num_features=2, momentum=0.5, eps=1e-05, affine=True)
    output = batch_norm(input_1d)
    print(f'output:{output}, output.shape:{output.shape}')


if __name__ == '__main__':
    batch_norm_1d_demo()


