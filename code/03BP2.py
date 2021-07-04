# 二层神经网络的反向传播示例
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0])
w1.requires_grad = True
w2 = torch.tensor([2.0])
w2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1 * (x ** 2) + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(100):
    print('epoch : ', epoch)
    for x, y in zip(x_data, y_data):
        loss_value = loss(x, y)
        loss_value.backward()
        print('\tgrad : ', 'w1: ', w1.grad.item(), 'w2:',  w2.grad.item(), 'b:', b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('loss_val: ', loss_value.item())
