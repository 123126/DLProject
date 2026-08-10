"""
思路：numpy对象 ->张量Tensor ->数据集对象TensorDataset->数据加载器DataLoader
"""

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号

def create_dataset():
    #1.创建数据集
    x, y, coef = make_regression(
        n_samples=100,      #100个样本
        n_features=1,       #一个特征
        noise=10,           #噪声
        coef=True,          #线性回归系数
        bias=14.5 ,         #偏置
        random_state=3      #随机数种子
    )

    #2.将上述数据集转换为张量
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y,dtype=torch.float)

    #3.返回结果
    return x,y,coef

#定义函数表示模型训练
def train(x, y , coef):
    #1.张量Tensor ->数据集对象TensorDataset->数据加载器DataLoader
    dataset = TensorDataset(x, y)

    #2.创建数据加载器对象
    #参1：数据集对象 ， 参2：批量大小， 参3：是否打乱数据（训练集打乱，测试集不打乱）
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    #3.创建模型
    #参1：输入特征数，参2：输出特征数
    model = torch.nn.Linear(in_features=1, out_features=1)

    #4.创建损失函数
    criterion = torch.nn.MSELoss()

    #5.创建优化器对象
    #参1：模型参数，参2：学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #6.训练模型具体过程
    #6.1 定义参数：训练轮次，每轮（平均损失），训练总损失，训练样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    #6.2 开始训练
    for epoch in range(epochs):   #分训练轮次
        for train_x , train_y in data_loader:  #分批次数据
            y_pred = model(train_x)
            #计算每批的平均损失值
            loss = criterion(y_pred, train_y.reshape(-1,1))
            total_loss += loss.item()
            total_sample += 1
            #梯度清零+反向传播+优化器更新参数
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        #把本轮平均损失值添加到列表中
        loss_list.append(total_loss/total_sample)
        print(f'第{epoch+1}轮，平均损失值：{total_loss/total_sample}')

    #7 打印最终训练结果
    print(f"{epoch}轮的平均损失值：{loss_list}")
    print(f"模型参数：{model.weight.data}, {model.bias.data}")

    #8.1绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('损失曲线变化图')
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.grid()
    plt.show()

    #9 绘画预测值和真实值的关系
    #9.1 绘制样本点的分布情况
    plt.scatter(x, y)
    #9.2 绘制模型预测值
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x], dtype=torch.float)
    #9.3 绘制真实值
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x], dtype=torch.float)
    plt.plot(x, y_pred, 'r-', label='预测值')
    plt.plot(x, y_true, 'b-', label='真实值')
    plt.legend()
    plt.title('真实值和预测值的关系')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    x,y,coef = create_dataset()
    # print(f'x:{x}, y:{y}, coef:{coef}')
    train(x, y, coef)
