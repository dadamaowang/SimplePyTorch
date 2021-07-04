# 04 PyTorch Fashion



## STEP1. Prepare Dataset

准备数据集



## STEP2. Design Model Using Class

设计模型

inherit from nn.Module



## STEP3. Construct Loss and Optimizer

构造损失函数和优化器



## STEP4.Training Circle

forward, backward, update



代码分段示例：

假设模型是$\hat{y}=\omega*x$

- 准备数据集：

```python3
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
```

x_data, y_data的三个样本都直接放进一个tensor里，一次能够计算出全部的结果

mini-batch风格

因此x和y都是3*1的矩阵

- 模型设计：

不再需要人为求导，重点在构造计算图

需要确认权重w和偏置b的维度，用输入和输出的维度来确认

特别注意：loss在计算图的结构中，计算出来的是一个tensor，我们需要的loss是一个标量

所以要对计算出的loss的tensor进行每个元素的求和 ,平均，得到一个标量



只要构造模型，都__必须__的元素：

__继承nn.Module__

__构造函数__

__forward__



【code】

```python
import torch


def main():
    # STEP 1 PREPARE DATA
    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[2.0], [4.0], [6.0]])
    # x_data和y_data都是矩阵形式
    # mini-batch风格
    # 这里是3行1列，共3个训练数据，每个数据1个特征
    # 在mini-batch里，一个输入矩阵X的行表示样本数目，列表示feature数目

    # STEP 2 MODULE
    # 所有的class都要继承nn.Module
    # 构造class的目的是搭建网络，实现forward
    # 至少要实现__init__和forward俩函数
    # __init__就是构造函数
    class LinearModule(torch.nn.Module):
        def __init__(self):
            super(LinearModule, self).__init__()  # 继承父类
            # 第一个参数是模型名称，这句super直接调用即可
            self.linear = torch.nn.Linear(1, 1)
            # nn.Linear是一个类
            # 相当于计算图中一部分Linear的构造
            # 作的运算就是线性运算 y = Ax + b
            # nn.Linear(in_features, out_features, bias=True)
            # x:n * in_features的矩阵，n个样本;y:n*out_features的矩阵，n个样本;
            # 目的是把合适的weight矩阵的维度拼出来
            # 实际运算时候可能转置

        def forward(self, x):  # 覆写了一个forward
            y_pred = self.linear(x)
            # linear(x)，典型的python风格的callable（可调用）的对象
            return y_pred

    model = LinearModule()  # 这个model就是可调用的

    # STEP3 CONSTRUCT LOSS AND OPTIMIZER
    # loss 为了反向传播
    # optimizer 为了更新梯度
    criterion = torch.nn.MSELoss(reduction='sum')
    # reduction是降维用的
    # 构造criterion对象计算MSELoss
    # 它自动就计算了MSE
    # criterion需要的参数就是y_pred和y

    # 优化器不是model
    # 在优化器里要告诉它具体哪些tensor是用来优化的
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # SGD是一个类，这里要实例化
    # 需要告诉优化器那些参数是要优化的
    # .parameter检查model里的所有成员，要优化就优化
    # 由于一个model里边，可能不止一个部分，最简单的方法就是.parameters一键优化所有的参数
    # 梯度下降/SGD/批量梯度下降，用的优化器API都是optim.SGD
    # lr学习率，这里简单地使用了统一学习率，pytorch里也支持对不同的部分使用不同的学习率

    # STEP4 TRAINING CYCLE (FORWARD, BACKWARD, UPDATE)
    for epoch in range(100):
        # forward
        y_pred = model(x_data)  # 前馈
        loss = criterion(y_pred, y_data)  # 算损失
        print('epoch : ', epoch, 'loss : ', loss)

        # backward
        # 训练前梯度归零，一定注意
        optimizer.zero_grad()
        loss.backward()  # 然后反向传播，更新

        # update : step函数用来更新模型的各个权重
        optimizer.step()
    print('finished.')
    print('w : ', model.linear.weight.item())
    print('b : ', model.linear.bias.item())


if __name__ == '__main__':
    main()
```



【result】

```
epoch :  94 loss :  0.16329249739646912
epoch :  95 loss :  0.1609458327293396
epoch :  96 loss :  0.15863269567489624
epoch :  97 loss :  0.15635278820991516
epoch :  98 loss :  0.1541057676076889
epoch :  99 loss :  0.15189111232757568
finished.
w :  1.7405472993850708
b :  0.5897968411445618
```

