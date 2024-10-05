import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt


class Network(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.layer1 = nn.Linear(1, m)
        self.layer2 = nn.Linear(m, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)


if __name__ == "__main__":
    # 初始化模型并打印结构
    model = Network(6)
    print(model)
    print("")

    # 打印参数的形状
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.shape}")
    print("")

    # 准备数据
    x = np.arange(0.0, 1.0, 0.01)
    y = np.sin(2 * np.pi * x)
    x = x.reshape(100, 1)
    y = y.reshape(100, 1)

    # 转换为Torch张量
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10000):
        h = model(x)  # 前向传播
        loss = criterion(h, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清空梯度

        if epoch % 1000 == 0:
            print(f"{epoch}, {loss.item()}")

    # 绘制结果
    h = model(x)
    plt.scatter(x.detach().numpy(), h.detach().numpy(), label='forcast')
    plt.scatter(x.detach().numpy(), y.detach().numpy(), label='real', alpha=0.5)
    plt.title("same")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
