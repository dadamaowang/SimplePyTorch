# 06 Multiple Dimension Model

处理多分类问题：

例子：糖尿病的数据，8个属性，一个输出（作为分类）

Logistic Regression Model:
$$
\hat{y}^{(i)} = \sigma(\sum_{n=1}^{8} x_n^{(i)}\cdot w_n + b )
$$
可以变成__向量化的运算__，利用硬件能力提高运算速度
$$
\begin{bmatrix} z^{(1)} \\. \\. \\. \\z^{(n)}\end {bmatrix} = 
\begin{bmatrix} x_1^{(1)} & . & . & . & x_8^{(1)}
                \\ . & . & . & . & .
                \\ . & . & . & . & .
                \\ . & . & . & . & .
                \\ x_1^{(n)} & . & . & . & x_8^{(n)}\end{bmatrix}
\begin{bmatrix} \omega_1 \\ . \\ . \\ . \\ \omega_8 \end{bmatrix} +
\begin{bmatrix} b \\. \\ . \\ . \\b \end{bmatrix}
$$


输入：X的矩阵，是N*8的；

输出：Z的矩阵，是N*1的

稍微改进Logistic 回归：

```
torch.nn.Linear(8, 1)
```

在构造一个多层的：（多个首尾相接，构成多层的神经网络）

比如8维转2维，等等线性变换；

有时候可能变换是非线性的，就需要通过多层的组合来实现这种变换

__神经网络本质上就是寻找一种非线性的空间变换函数__

网络越复杂，层数越多，学习能力往往越强，但不是越强越好；

比如数据中有噪声的时候，网络会把噪声也学到，我们想要的只是对真值的学习，所以网络的__泛化能力__很重要



重要的能力：__读文档__;  __基本架构理念（CPU,主机，操作系统...）__  ; __泛化能力__



网络结构：

![MultipleDimension01](C:\shelf\Projects\PytorchLeadIn\pics\MultipleDimension01.PNG)



【code:心脏病数据分类器】

```python
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 准备数据
    xy = np.loadtxt('diabete_data.csv', delimiter=',', dtype=np.float32, encoding='UTF-8-sig')
    print(xy.shape)
    # xy中，包含768条样本，每个样本的前8列是属性值，最后一列是0或1，代表心脏病的分类
    x_data = torch.from_numpy(xy[:, :-1])  # x_data从xy中提取出全部样本的属性
    y_data = torch.from_numpy(xy[:, [-1]])  # y_data从xy中提取出分的类，确认是矩阵形式不是数组形式

    class DiabeteClassifier(torch.nn.Module):
        def __init__(self):
            super(DiabeteClassifier, self).__init__()
            self.linear1 = torch.nn.Linear(8, 6)  # 降维1：输入N*8->N*6
            self.linear2 = torch.nn.Linear(6, 4)
            self.linear3 = torch.nn.Linear(4, 1)
            self.sigmoid = torch.nn.Sigmoid()  # sigmoid是网络的一层，没有参数要训练

        def forward(self, x):
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            x = self.sigmoid(self.linear3(x))
            return x

    my_model = DiabeteClassifier()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1)

    epoch_list = []
    loss_list = []
    for epoch in range(100):
        y_pred = my_model(x_data)
        loss_val = criterion(y_pred, y_data)
        print('epoch:', epoch, 'loss:', loss_val.item())
        epoch_list.append(epoch)
        loss_list.append(loss_val)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('08multi.png')
    plt.show()


if __name__ == '__main__':
    main()

```

【result】

```
epoch: 89 loss: 0.6455753445625305
epoch: 90 loss: 0.6455677151679993
epoch: 91 loss: 0.6455603241920471
epoch: 92 loss: 0.6455529928207397
epoch: 93 loss: 0.6455457210540771
epoch: 94 loss: 0.6455386877059937
epoch: 95 loss: 0.6455316543579102
epoch: 96 loss: 0.6455247402191162
epoch: 97 loss: 0.6455178260803223
epoch: 98 loss: 0.6455109715461731
epoch: 99 loss: 0.6455042362213135
```

![MultipleDimension02](C:\shelf\Projects\PytorchLeadIn\pics\MultipleDimension02.png)



