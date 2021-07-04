# 05 Logistic Regression and Sigmoid Functions



线性回归中，需要估计的估计值$\hat{y}$属于一个连续的空间，类似这样的任务叫做回归任务

有时候需要的是分类问题，$\hat{y}$的值的集合是离散的

比如二分类问题，最终计算的就是$\hat{y}=1$的概率和$\hat{y}=0$的概率

由于上述两个概率相加是1，这里结果只输出一个值就可以



为了使分类器的计算结果__映射成输出结果的0或1__，选取logistic函数：
$$
y = \frac{1}{1+e^{-x}}
$$
当$x \to +\infty$时，$y \to 1$; $x \to -\infty$时，$y \to 0$;

函数的特点：值超过某一阈值之后，导数的值变的非常小（饱和函数）

作分类的时候可以那线性模型，把输出用logistic函数处理一下

除了logistic函数，还有很多sigmoid型函数（最典型的sigmoid就是logistic)

logistic regression函数的模型：
$$
\hat{y}=\sigma(\omega*x + b)
$$


损失函数：

原来的loss:
$$
Loss = (\hat{y}-y)^2
$$
由于输出的是个分布，几何距离上的loss已经不再适用

新的loss:
$$
loss = -(y\log{\hat{y}} + (1-y)\log{(1-\hat{y})})
$$
对二分类，需要比较的是两个分布的差异

对上述公式，把y=0和y=1代入，化简，可以知道$\hat{y}$越接近真实的分类，loss会越小；



Mini-Batch Loss Function for Binary Classification:
$$
loss = -\frac{1}{N}\sum_{n=1}^{N}{y_n\log\hat{y_n} + (1-y_n)\log{1-\hat{y_n}}}
$$


小批量地对一个batch的样本求loss的均值

- code

```python
# 和pytorch fashion的基本框架一致
# 就是training set变成了分类
# 然后loss变成了计算BCE，加了sigmoid函数平滑用

import torch


def main():

    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[0.0], [0.0], [1.0]])

    class LogisticRegressionModel(torch.nn.Module):
        def __init__(self):
            super(LogisticRegressionModel, self).__init__()
            self.linear = torch.nn.Linear(1, 1)
            # 由于sigmoid函数没有参数，所以不需要在初始化的代码里改变什么

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

    model = LogisticRegressionModel()
    criterion = torch.nn.BCELoss(reduction='sum')
    # size_average就是求不求均值的问题，也就是loss的导数计算里边前边还有没有系数
    # 对是否计算均值（添加一个系数）,影响的是学习率的选择
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        y_pred_val = model(x_data)
        loss_val = criterion(y_pred_val, y_data)
        print('epoch: ', epoch, 'loss:', loss_val.item())

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    print('final:')
    print('w:', model.linear.weight.item())
    print('b:', model.linear.weight.item())


if __name__ == '__main__':
    main()

```

- result

```
epoch:  1 loss: 1.8130683898925781
epoch:  2 loss: 1.804075002670288
epoch:  3 loss: 1.7956411838531494
epoch:  4 loss: 1.7877299785614014
epoch:  5 loss: 1.7803049087524414
epoch:  6 loss: 1.7733328342437744
epoch:  7 loss: 1.7667815685272217
epoch:  8 loss: 1.7606217861175537
epoch:  9 loss: 1.7548251152038574
final:
w: 0.1745123416185379
b: 0.1745123416185379


```

