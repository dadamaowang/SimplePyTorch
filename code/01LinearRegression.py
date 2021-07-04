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
# 要是想保存图片必须先存后显示
plt.savefig(r'C:\shelf\CV_study\pytorch_study\notes\simple_linear1.png')
plt.show()


# deep learning 的可视化和打印日志很重要
# 观察模型训练的情况和程度





