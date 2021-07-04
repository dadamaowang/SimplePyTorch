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
