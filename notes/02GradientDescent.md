# 02 Gradient Descent

梯度下降法本质：找到使损失函数下降最快的方向， 并沿着这个方向更新权值，从而找到最佳权值



from initial guess to global cost minimum

损失函数对权值求导，得到梯度（正向）

更新权重
$$
\omega = \omega - \alpha\frac{\partial cost}{\partial\omega}
$$
其中，$\alpha$是学习率

In this file, we do the gradient descent manually 

【code】

```python
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 猜测初始w


def forward(x):
    return w * x


# 计算mse
def cost(xc, yc):
    cost1 = 0
    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        cost1 += (y_pred - y) ** 2
    return cost1 / len(xc)


# compute gradient
# 也是求和再求平均值
# 求得的是对一次猜测w，全部训练样本得到的梯度的平均值
def gradient(xc, yc):
    grad = 0
    for x, y in zip(x_data, y_data):
        grad += 2 * x * (x * w - y)
    return grad / len(xc)


# 训练过程 : 每次都要拿权重w减去学习率乘以grad
epoch_list = []  # 训练迭代次数
cost_list = []   # 损失
for epoch in range(100):  # 迭代训练100次
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # learning rate is 0.01
    print('Epoch : ', epoch, 'cost_val :', cost_val, 'w : ', w)
    epoch_list.append(epoch)
    cost_list.append(cost_val)


plt.plot(epoch_list, cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.savefig(r'C:\shelf\CV_study\pytorch_study\notes\03gradient_descent.png')
plt.show()


```

【result】

```
Epoch :  92 cost_val : 6.908348768398496e-08 w :  1.9998896858085284
Epoch :  93 cost_val : 5.678969725349543e-08 w :  1.9998999817997325
Epoch :  94 cost_val : 4.66836551287917e-08 w :  1.9999093168317574
Epoch :  95 cost_val : 3.8376039345125727e-08 w :  1.9999177805941268
Epoch :  96 cost_val : 3.154680994333735e-08 w :  1.9999254544053418
Epoch :  97 cost_val : 2.593287985380858e-08 w :  1.9999324119941766
Epoch :  98 cost_val : 2.131797981222471e-08 w :  1.9999387202080534
Epoch :  99 cost_val : 1.752432687141379e-08 w :  1.9999444396553017

```

![GradientDescent01](C:\shelf\Projects\PytorchLeadIn\pics\GradientDescent01.png)

