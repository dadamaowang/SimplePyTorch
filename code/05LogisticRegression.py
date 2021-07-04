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
