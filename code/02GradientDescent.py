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





