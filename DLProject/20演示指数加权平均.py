"""
采用计算近30天天气温度例子，演示指数加权平均效果
β越大，表示数据预测越依赖历史数据，折线图越平缓
β越小，表示数据预测越依赖当前数据,折线图越陡峭
"""
import torch
import matplotlib.pyplot as plt

ELEMENT_NUM = 30

def demo1(): 
    torch.manual_seed(0)
    p_true = torch.randn(ELEMENT_NUM, 1)
    days = torch.arange(1, ELEMENT_NUM+1, 1)
    plt.plot(days, p_true, 'r-', label='true')
    plt.scatter(days, p_true)
    plt.show()

def demo2(beta=0.9):
    torch.manual_seed(0)
    p_true = torch.randn(ELEMENT_NUM, 1)
    exp_weight_avg = []
    for idx, temp in enumerate(p_true, 1):
        if idx == 1:
            exp_weight_avg.append(temp)
        else:
            exp_weight_avg.append(beta * exp_weight_avg[-1] + (1-beta) * temp)
    days = torch.arange(1, ELEMENT_NUM + 1, 1)
    plt.plot(days, exp_weight_avg, 'r-', label='true')
    plt.scatter(days, p_true)
    plt.show()


if __name__ == '__main__':
    demo1()
    demo2(0.9)
    demo2(0.5)

