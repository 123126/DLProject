"""
卷积神经网络介绍:
    概念:
        全称叫:Convolutional neural network，即:包含卷积层的神经网络.
    组成:
        卷积层(Convolutional):用于提取图像的 局部特征，结合 卷积核(每个卷积核=1个神经元)实现，处理后的结果叫:特征图.
        池化层(Pooling):用于 降维，降采样
        全连接层(Full Connected,fc,linear):用于 预测结果，并输出结果的.
    特征图计算方式:
        N=(W-F+2xP)/S +1
        W:输入图像的大小F:卷积核的大小
        P:填充的大小
        S:步长
        N:输出图像的大小(特征图大小)
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demo():
    img = plt.imread('./data/test.png')
    img = torch.tensor(img, dtype=torch.float32)
    print(f'img.shape:{img.shape}')
    #HWC->CHW
    img2 = torch.permute(img, (2, 0, 1))
    print(f'img2.shape:{img2.shape}')

    #增加一个维度->(1, 3, 1039, 1045)
    img3 = torch.unsqueeze(img2, dim=0)
    print(f'img3.shape:{img3.shape}')

    #卷积运算 参1:输入通道数 参2:输出通道数 参3:卷积核大小 参4:步长 参5:填充
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)
    img4 =conv(img3)
    print(f'img4.shape:{img4.shape}')

    #CHW->HWC
    img5 = torch.permute(img4, (0, 2, 3, 1)).detach().numpy()
    plt.imshow(img5[0, :, :, 1])
    plt.show()


if __name__ == '__main__':
    demo()
