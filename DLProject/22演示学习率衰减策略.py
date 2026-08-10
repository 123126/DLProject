"""
学习率衰减策略介绍:
    目的:
        较之于AdaGrad，RMSProp，Adam方式，我们可以通过 等间隔，指定间隔，指数等方式，来手动控制学习率的调整.
    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减
    等间隔学习率衰减:
        step_size:  间隔的轮数，即:多少轮调整一次学习率.
        gamma:      学习率衰减系数，即:Lr新=Lr旧* gamma

    指定间隔学习率衰减:
        milestones = [50, 125, 160]里边定义的是要调整学习率的 轮数.
        gamma:学习率衰减系数，即:Lr新=Lr旧* gamma

    指数间隔学习率衰减:
        前期学习率衰减快，中期慢，后期更慢，更符合梯度下降规律.
        公式:
            Lr新 = Lr/旧 * gamma ** epoch
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 1.演示等间隔学习率衰减
def demo1():
    # 定义参数: 训练轮数，批数，初始学习率
    epochs, batch_size, lr = 200, 10, 0.1
    y_true = torch.tensor([0], dtype=torch.float)
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], dtype=torch.float, requires_grad=True)

    # 创建优化器，使用动量法
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 参1：优化器， 参2：指定间隔轮数， 参3：学习率衰减参数
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    lr_list, epochs_list = [], []
    for epoch in range(epochs):
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率

        for i in range(batch_size):
            y_pred = w * x
            loss = (y_true - y_pred) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        # 更新学习率
        scheduler.step()

    # 可视化
    plt.plot(epochs_list, lr_list)
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()

# 2.演示指定间隔学习率衰减
def demo2():
    # 定义参数: 训练轮数，批数，初始学习率
    epochs, batch_size, lr = 200, 10, 0.1
    y_true = torch.tensor([0], dtype=torch.float)
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], dtype=torch.float, requires_grad=True)

    # 创建优化器，使用动量法
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 参1：优化器， 参2：指定间隔轮数， 参3：学习率衰减参数
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=0.5)

    lr_list, epochs_list = [], []
    for epoch in range(epochs):
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率

        for i in range(batch_size):
            y_pred = w * x
            loss = (y_true - y_pred) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        # 更新学习率
        scheduler.step()

    # 可视化
    plt.plot(epochs_list, lr_list)
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()


# 3.演示指数学习率衰减
def demo3():
    # 定义参数: 训练轮数，批数，初始学习率
    epochs, batch_size, lr = 200, 10, 0.1
    y_true = torch.tensor([0], dtype=torch.float)
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], dtype=torch.float, requires_grad=True)

    # 创建优化器，使用动量法
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 参1：优化器， 参2：指定间隔轮数， 参3：学习率衰减参数
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    lr_list, epochs_list = [], []
    for epoch in range(epochs):
        epochs_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率

        for i in range(batch_size):
            y_pred = w * x
            loss = (y_true - y_pred) ** 2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        # 更新学习率
        scheduler.step()

    # 可视化
    plt.plot(epochs_list, lr_list)
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.show()



if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
