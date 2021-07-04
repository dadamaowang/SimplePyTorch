# 改进08的处理多维特征输入
# mini_batch fashion

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def main():

    class DiabetesData(Dataset):
        def __init__(self, filepath):
            xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, encoding='UTF-8-sig')
            # 这里数据集要是比较大，初始化不要直接加载进内存里
            self.len = xy.shape[0]
            # shape 是元组(n,9)这里就能得到n，后边len方法就方便
            self.x_data = torch.from_numpy(xy[:, :-1])
            self.y_data = torch.from_numpy(xy[:, [-1]])

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
            # 因为是直接加载进内存里边，所以直接按index返回就可以

        def __len__(self):
            return self.len

    my_dataset = DiabetesData('diabete_data.csv')  # 构造数据对象
    my_train_loader = DataLoader(dataset=my_dataset, batch_size=32, shuffle=True, num_workers=0)
    # windows CPU 下num_workers只能是0

    class DiabeteClassifier(torch.nn.Module):
        def __init__(self):
            super(DiabeteClassifier, self).__init__()
            self.linear1 = torch.nn.Linear(8, 6)
            self.linear2 = torch.nn.Linear(6, 4)
            self.linear3 = torch.nn.Linear(4, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            x = self.sigmoid(self.linear3(x))
            return x

    my_model = DiabeteClassifier()
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1)

    for epoch in range(100):
        for i, data in enumerate(my_train_loader, 0):
            inputs, labels = data
            y_pred = my_model(inputs)
            loss_val = criterion(y_pred, labels)

            print('epoch:', epoch, 'batch:', i, 'loss:', loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()


if __name__ == '__main__':
    main()








