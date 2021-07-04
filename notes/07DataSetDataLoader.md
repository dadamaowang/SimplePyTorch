# 07 DataSet & DataLoader



Dataset: 索引

DataLodaser: 拿出mini-batch



mini-batch训练方法：

```python3
for epoch in range(train_epoches):
    for i in range(total_batch):
```

最外层是epoch的循环；

每个epoch内一个batch一个batch的取出来训练（所有的batch都会被跑一遍）



***

关于epoch, batch 和 iteration:

epoch: one forward pass and one backward pass for __all the training examples__

batch_size: __the number of training examples__ in one forward backward pass

iteration: number of passes, each pass using __batch-size__ number of examples

eg. 10000个样本；batch-size=1000；则iteration=10

****



DataLoader 的重要参数： batch-size, shuffle(提高数据样本的随机性)

DataLoader:只要数据集能够：根据索引访问数据元素；知道数据的总长度(len)

那么它就能自动地小批量地取出数据打乱，也就是做成一个可迭代的loader，一次一次取出来

就可以用for, in 的循环把dataloader里边所有的batch都拿出来



代码层面实现dataset和dataloader:

```python
import torch
from torch.utils.data import Dataset  # Dataset：抽象类，自己覆写里边的方法
from torch.utils.data import DataLoader  # 可以实例化一个DataLoader来帮助加载数据

class MyDataset(Dataset):  # Dataset不能实例化，只能被子类继承
    def  __init__(self):
        pass
    
    def __getitem__(self, index):
        pass     
    # getitem可以根据下标（索引）拿数据，是个魔法方法
    
    def __len__(self):
        pass 
    # 可以返回数据条数

my_data = MyDataset()  # 实例化一个
my_train_loader = DataLoader(
                  dataset=my_data,  # 因为要的肯定是my_data的loader
                  batch_size=32,
                  shuffle=True,
                  num_workers=2  # 这个就是读batch的时候是否要并行线程，要的话要几个，可以提高读取效率
                  )
# ...

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
```

notice:

DataLoader里边的init有两种方式：

第一种是直接把所有数据拿出来放内存里，完了后边想要就随便拿；（适用于数据集本身量就不大）

[结构化数据，关系表]

第二种是放文件里，只把索引加载进来，需要谁现去文件里拿；

[图像/语音等非结构化数据]



训练过程的改变：

```python
for epoch in range(100):  # 二重循环：所有数据都跑100遍
    for i, data in enumerate(my_train_loader, 0):  # 对my_train_loader作迭代
    # 用enumerate是为了获得是第几次迭代
        inputs, labels = data  # 从train_loader里拿出来的元组放到了data里边
        # 注意这里的inputs和labels对应训练样本的x和y，
        # loader拿出来一个batch的x和对应的y后，
        # 将x和y分别变成矩阵，然后自动转换成tensor
        # 所以这里的inputs和labels都是tensor
        
```



关于torchvision.datasets:

包含了很多上述dataset的子类，所以有getitem和len方法，还支持多线程，算是预处理好的可以直接使用的数据集的类

一般要是直接用的话，直接from torchvision import datasets,然后填几个参数自己构造就行

比如MNIST:

```python
import torch
from torchvision import dateset

my_mnist = datasets.MNIST(root='',
                          train=True,   # 是训练的
                          transform=my_transform  # 自定义的变换
                          download=True,  # 是不是要下载
                          )
# 构造test_my_mnist
# 构造train_loader
# 构造test_loader

```



【code:】

```python
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
    # 数据与第六次的一样，是768条心脏病数据，batch_size为32，则每一个epoch的iteration为24次
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
            print('epoch:', epoch, 'loss:', loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()


if __name__ == '__main__':
    main()


```

【result:】

可以看到每个epoch里，都训练了24个batch,一个batch的size为32，32*24=768,为全部训练样本的个数

```python3
epoch: 98 batch: 21 loss: 0.619433581829071
epoch: 98 batch: 22 loss: 0.6449031233787537
epoch: 98 batch: 23 loss: 0.706240177154541
epoch: 99 batch: 0 loss: 0.6286807656288147
epoch: 99 batch: 1 loss: 0.5764321684837341
epoch: 99 batch: 2 loss: 0.8026859164237976
epoch: 99 batch: 3 loss: 0.5483067035675049
epoch: 99 batch: 4 loss: 0.48950278759002686
epoch: 99 batch: 5 loss: 0.6750475168228149
epoch: 99 batch: 6 loss: 0.6441752910614014
epoch: 99 batch: 7 loss: 0.6052690744400024
epoch: 99 batch: 8 loss: 0.5358450412750244
epoch: 99 batch: 9 loss: 0.6149870157241821
epoch: 99 batch: 10 loss: 0.6763702630996704
epoch: 99 batch: 11 loss: 0.6888394951820374
epoch: 99 batch: 12 loss: 0.5518947839736938
epoch: 99 batch: 13 loss: 0.45255208015441895
epoch: 99 batch: 14 loss: 0.7021615505218506
epoch: 99 batch: 15 loss: 0.6079815030097961
epoch: 99 batch: 16 loss: 0.6692335605621338
epoch: 99 batch: 17 loss: 0.649067759513855
epoch: 99 batch: 18 loss: 0.5493977069854736
epoch: 99 batch: 19 loss: 0.5313356518745422
epoch: 99 batch: 20 loss: 0.5696055293083191
epoch: 99 batch: 21 loss: 0.5724165439605713
epoch: 99 batch: 22 loss: 0.62577223777771
epoch: 99 batch: 23 loss: 0.7282902598381042

```

debug观察my_train_loader和my_dataset,可以看到batch_size是32,前边itertion是24

![DD01](C:\shelf\Projects\PytorchLeadIn\pics\DD01.PNG)

