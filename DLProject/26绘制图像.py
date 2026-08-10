"""
图像分类:
    二值图:    1通道，每个像素点由0，1组成
    灰度图:    1通道，每个像素点的范围:[0，255]
    索引图:    1通道，每个像素点的范围:[0，255]，像素点表示颜色表的索引
    RGB真彩图: 3通道，Red,Green,Blue,红绿蓝.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np


#绘制全黑图像
def demo1():
    #HWC
    img = np.zeros((100, 100, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img2 = torch.full((100, 100, 3), 255)
    plt.axis("off")
    plt.imshow(img2)
    plt.show()

#
def demo2():
    # 1.加载图片
    img = plt.imread("./data/test.png")
    print(f"img:{ img},\n img.shape:{img.shape}")

    # 2.展示图片
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # 3.保存图片
    plt.imsave(r'./data/test_copy.png', img)


if __name__ == '__main__':
    # demo1()
    demo2()
