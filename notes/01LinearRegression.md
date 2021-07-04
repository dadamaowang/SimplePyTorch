# 01 Linear Regression



最最简单的线性模型：
$$
y = \omega \times x + b
$$
其中，$y$是目标，$x$是样本，$\omega$和$b$是模型的参数

寻找使损失函数达到全局最小值时的$\omega$和$b$

适用于简单的线性预测问题，二分类问题等等



## General Steps:

Pytorch general step of making and training a model：

__prepare data -> select model -> train(weight) __



为了更加简化，选择仅有一个参数$\omega$的线性模型：
$$
\hat{y} = x \ast \omega
$$

目的：预测$\omega$ 

步骤：the machine starts with a random guess $\omega$，然后对每个$\omega$进行评估；

计算$\hat{y_1}$,$\hat{y_2}$,......（预测值）；

找到的模型的预测值和真实值的差（loss,损失）；

针对每一条样本计算它的loss；

这里以最小二乘计算loss为例：
$$
Loss = (\hat{y}-y)^2 = (x*\omega-y)^2
$$


针对每个样本计算完loss后，常用MSE计算平均loss,想办法让average loss降到最低;

(loss是平方项，必大于0，而理想状态平均loss为0，显然不可能，就是想找让loss最小的)

__平均平方误差 Mean Square Error: (MSE)__
$$
cost = \frac{1}{N}\sum_{n=1}^{N}(\hat{y}-y)^2
$$
(对不同的$\omega$,计算出的MSE不同。)













```python
# 一个简单的线性模型训练
# y = w * x, 训练w

# STEP.0导入需要的模块
import numpy as np
import matplotlib.pyplot as plt

# STEP.1导入数据
# 设函数为 y = 2x
x_data = [1.0, 2.0, 3.0]  # 训练用数据
y_data = [2.0, 4.0, 6.0]
# 一般训练时，X，Y分开存储
# 这里只有简单的3个样本

# STEP.3定义模型：
def forward(x):  # 一个简单的线性模型
    return x * w  # w的值是从for循环中传入的
# 模型要训练w

# STEP.4定义损失函数
def loss(x, y):
    y_pred = forward(x)  # y_pred是模型预测出来的y
    return (y_pred - y) ** 2  # 返回的就是y_pred减去真实y的平方

# STEP.5训练
# 先做训练的准备工作，因为要取很多个w并计算它们的MSE,两个空list作为保存用
w_list = []  # 后边画出图的横坐标，先列个表
mse_list = []  # 后边画出图的纵坐标，先列个表

for w in np.arange(0.0, 4.1, 0.1):  # 用arrange生成w:0.0， 0.1， 0.2，...
# 这个最外层的for循环可以看作是'epoch'，训练轮数
    print('w = ', w)
    loss_sum = 0  # 每轮的开始,loss的总和是0
    w_list.append(w)  # 本轮的w加入进w_list
    for x_val, y_val in zip(x_data, y_data):
        # python的zip函数将x_data,y_data逐个取出，拼成元组
        # 对每一个样本：
        y_pred_val = forward(x_val)  # 先将x放进模型计算y_pred
        loss_val = loss(x_val, y_val)  # 再计算loss
        loss_sum += loss_val  # 将这个样本对应的loss累计起来
        # 打印x,y,预测的y和loss
        print('x_val:', x_val, '\t', 'y_val:', y_val)
        print('y_pred_val:', y_pred_val, '\t', 'loss_val:', loss_val)
    # for循环结束，这个w训练完了
    mse_list.append(loss_sum / 3)  # 求个loss均值放进mse_list里(除3是因为有3个样本)

# STEP.6总结训练结果
plt.plot(w_list, mse_list)  # 以w为横坐标，mse_list为纵坐标画图
plt.xlabel('w')
plt.ylabel('Loss')
plt.show()


# deep learning 的可视化和打印日志很重要
# 观察模型训练的情况和程度


```

【 result】

```
w =  1.2000000000000002
x_val: 1.0 	 y_val: 2.0
y_pred_val: 1.2000000000000002 	 loss_val: 0.6399999999999997
x_val: 2.0 	 y_val: 4.0
y_pred_val: 2.4000000000000004 	 loss_val: 2.5599999999999987
x_val: 3.0 	 y_val: 6.0
y_pred_val: 3.6000000000000005 	 loss_val: 5.759999999999997
w =  1.3
x_val: 1.0 	 y_val: 2.0
y_pred_val: 1.3 	 loss_val: 0.48999999999999994
x_val: 2.0 	 y_val: 4.0
y_pred_val: 2.6 	 loss_val: 1.9599999999999997
x_val: 3.0 	 y_val: 6.0
y_pred_val: 3.9000000000000004 	 loss_val: 4.409999999999998
w =  1.4000000000000001
x_val: 1.0 	 y_val: 2.0
y_pred_val: 1.4000000000000001 	 loss_val: 0.3599999999999998
x_val: 2.0 	 y_val: 4.0
y_pred_val: 2.8000000000000003 	 loss_val: 1.4399999999999993
x_val: 3.0 	 y_val: 6.0
y_pred_val: 4.2 	 loss_val: 3.2399999999999993
w =  1.5
x_val: 1.0 	 y_val: 2.0
y_pred_val: 1.5 	 loss_val: 0.25
x_val: 2.0 	 y_val: 4.0
y_pred_val: 3.0 	 loss_val: 1.0
x_val: 3.0 	 y_val: 6.0
y_pred_val: 4.5 	 loss_val: 2.25

```

![LR01](C:\shelf\Projects\PytorchLeadIn\pics\LR01.png)





