# -*- coding: utf-8 -*-
# 实验2：螺旋数据分类 (Spiral Classification)
# 对应实验指导 https://oucaigroup.feishu.cn/wiki/Nbddwe54fiolLFkoslfcdYcAnZe
# 本地环境：CPU 版 PyTorch（无 GPU 时自动用 cpu）

import random
import math

import torch
from torch import nn, optim
from IPython import display

from plot_lib import plot_data, plot_model, set_default

# ============ 初始化 ============
# 本地没有 GPU 时自动回退到 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# 设置随机数种子，保证结果可复现
seed = 12345
random.seed(seed)
torch.manual_seed(seed)

N = 1000   # 每类样本数量
D = 2      # 每个样本的特征维度
C = 3      # 样本类别数
H = 100    # 神经网络隐层单元数量

# ============ 生成螺旋数据 ============
X = torch.zeros(N * C, D).to(device)
Y = torch.zeros(N * C, dtype=torch.long).to(device)
for c in range(C):
    index = 0
    t = torch.linspace(0, 1, N)  # 在 [0,1] 间均匀取 N 个数
    # 三类样本的角度区间相互交错，构成螺旋形分布，并加入少量噪声
    inner_var = torch.linspace((2 * math.pi / C) * c,
                               (2 * math.pi / C) * (2 + c), N) + torch.randn(N) * 0.2
    for ix in range(N * c, N * (c + 1)):
        X[ix] = t[index] * torch.FloatTensor(
            (math.sin(inner_var[index]), math.cos(inner_var[index])))
        Y[ix] = c
        index += 1

print("Shapes:")
print("X:", X.size())
print("Y:", Y.size())

# 可视化训练数据（螺旋形）
set_default()
plot_data(X, Y)


# ============ 1. 线性模型分类（无激活函数） ============
def train_linear():
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    # 两层线性模型，层间没有激活函数
    model = nn.Sequential(
        nn.Linear(D, H),
        nn.Linear(H, C)
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                weight_decay=lambda_l2)

    for t in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, Y)
        score, predicted = torch.max(y_pred, 1)
        acc = (Y == predicted).sum().float() / len(Y)
        print('[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f' % (t, loss.item(), acc))
        display.clear_output(wait=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 查看预测输出结构与含义
    print('y_pred.shape:', y_pred.shape)   # [3000, 3]，每个样本 3 个类别的得分
    print('y_pred[10, :]:', y_pred[10, :])
    print('score[10]:', score[10])
    print('predicted[10]:', predicted[10])

    # 绘制训练后的模型决策边界
    print(model)
    plot_model(X, Y, model)


# ============ 2. 两层神经网络分类（加入 ReLU 激活函数） ============
def train_nn():
    learning_rate = 1e-3
    lambda_l2 = 1e-5
    # 与线性模型相比，两层之间加入了 ReLU 激活函数
    model = nn.Sequential(
        nn.Linear(D, H),
        nn.ReLU(),
        nn.Linear(H, C)
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=lambda_l2)  # 内置 L2 正则

    for t in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, Y)
        score, predicted = torch.max(y_pred, 1)
        acc = (Y == predicted).sum().float() / len(Y)
        print("[EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), acc))
        display.clear_output(wait=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model)
    plot_model(X, Y, model)


if __name__ == '__main__':
    print('\n===== 线性模型（无激活函数）=====')
    train_linear()
    print('\n===== 两层神经网络（ReLU）=====')
    train_nn()
