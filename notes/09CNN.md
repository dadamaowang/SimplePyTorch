# 09CNN

## basic CNN

非常简单的串行结构

前边的卷积层和池化层统称特征提取（Feature Extraction）



注意左上角是图像的坐标原点（0，0）

卷积的特定权重确定后，就能从图像中提取特定的特征

__每一个通道配一个卷积核__

直接卷积最后得到的通道只有一个

如果要输出的是多个通道，输出的通道有几个，就要几个卷积



卷积层维度说明

![CNN01](C:\shelf\Projects\PytorchLeadIn\pics\CNN01.PNG)



计算示例：

![CNN02](C:\shelf\Projects\PytorchLeadIn\pics\CNN02.PNG)

conv_layer的size [10, 5, 3, 3]就是对应的卷积的m(输出的通道数)， n(输入的通道数)， kernel_weight, kernel_height



kernel_size默认是等于stride的



【code : 简单CNN】

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as ff
import torch.optim as optim


def main():
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ])
    train_dataset = datasets.MNIST(root='../datasets/mnist/', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../datasets/mnist/', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            self.pooling = torch.nn.MaxPool2d(2)
            self.fc = torch.nn.Linear(320, 10)

        def forward(self, x):
            fc_size = x.size(0)
            x = ff.relu(self.pooling(self.conv1(x)))
            x = ff.relu(self.pooling(self.conv2(x)))
            x = x.view(fc_size, -1)  # 此处算出fc层输入是320
            x = self.fc(x)
            return x

    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = criterion(outputs, target)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()
            if batch_idx % 300 == 299:
                print('[%d, %5d] loss : %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    main()
```

【result】

```
[1,   300] loss : 0.640
[1,   600] loss : 0.197
[1,   900] loss : 0.143
[2,   300] loss : 0.117
[2,   600] loss : 0.098
[2,   900] loss : 0.088
[3,   300] loss : 0.080
[3,   600] loss : 0.072
[3,   900] loss : 0.071
[4,   300] loss : 0.063
[4,   600] loss : 0.061
[4,   900] loss : 0.059
[5,   300] loss : 0.055
[5,   600] loss : 0.052
[5,   900] loss : 0.052
[6,   300] loss : 0.046
[6,   600] loss : 0.048
[6,   900] loss : 0.050
[7,   300] loss : 0.043
[7,   600] loss : 0.046
[7,   900] loss : 0.041
[8,   300] loss : 0.040
[8,   600] loss : 0.040
[8,   900] loss : 0.041
[9,   300] loss : 0.037
[9,   600] loss : 0.038
[9,   900] loss : 0.039
[10,   300] loss : 0.034
[10,   600] loss : 0.035
[10,   900] loss : 0.034
accuracy on test set: 98 %



```



## Advanced CNN

复杂的CNN不仅仅是串行结构，还会有并行，输出返回输入，等等很多结构上的改进

### GoogLeNet 

![CNN03](C:\shelf\Projects\PytorchLeadIn\pics\CNN03.PNG)

定义了一个一个地Inception块

Inception块：封装起来，减少代码冗余

关于Inception Module: 设计初衷：因为一般的卷积不知道Kernel取多大比较好，就放进一个Inception里边，里边有好几个Kernel，然后看哪个好，哪个Kernel的权重就大一点

![CNN05](C:\shelf\Projects\PytorchLeadIn\pics\CNN05.PNG)

Concatenate: 将得到的Tensor沿着Channel这个维度拼起来

得到的tensor的Height和Width必须一致

1*1卷积的作用：

![CNN06](C:\shelf\Projects\PytorchLeadIn\pics\CNN06.PNG)

节省计算量（权重少）

1*1卷积:Network in network

![CNN07](C:\shelf\Projects\PytorchLeadIn\pics\CNN07.PNG)

上面带有self的是init里边的，网络结构部分

下边的是forward里边的，是具体的计算部分

最后沿着dim=1这个维度（Channel）合并到一起去

![CNN08](C:\shelf\Projects\PytorchLeadIn\pics\CNN08.PNG)

【code:简易GoogLeNet】

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


def main():
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../datasets/mnist/', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../datasets/mnist/', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    class InceptionA(nn.Module):
        def __init__(self, in_channels):
            super(InceptionA, self).__init__()
            self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

            self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
            self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

            self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
            self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
            self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

            self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        def forward(self, x):
            branch1x1 = self.branch1x1(x)

            branch5x5 = self.branch5x5_1(x)
            branch5x5 = self.branch5x5_2(branch5x5)

            branch3x3 = self.branch3x3_1(x)
            branch3x3 = self.branch3x3_2(branch3x3)
            branch3x3 = self.branch3x3_3(branch3x3)

            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            branch_pool = self.branch_pool(branch_pool)

            outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
            return torch.cat(outputs, dim=1)
            # [B, C, W, H] 沿着C这个维度（沿着C这个方向拼接的）
            # InceptionA的out_channels：24*3+16 = 88

    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
            # 这里的88是因为后边x先经过conv1然后经过了InceptionA

            self.incep1 = InceptionA(in_channels=10)
            self.incep2 = InceptionA(in_channels=20)

            self.mp = nn.MaxPool2d(2)
            self.fc = nn.Linear(1408, 10)
            # 最后这个1408事先未知，可以通过先将程序跑到Linear之前，观察输出的size来得到
            # （人肉算也行，这样子保证准确）

        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.mp(self.conv1(x)))
            x = self.incep1(x)
            x = F.relu(self.mp(self.conv2(x)))
            x = self.incep2(x)
            # 下边两行可以在先跑完前边的再加上，这样上边的1408就能先知道了
            x = x.view(in_size, -1)
            x = self.fc(x)
            return x

    model = MyNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = criterion(outputs, target)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()
            if batch_idx % 300 == 299:
                print('[%d, %5d] loss : %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    main()


```

【result:】

```
[1,   300] loss : 0.839
[1,   600] loss : 0.185
[1,   900] loss : 0.136
[2,   300] loss : 0.110
[2,   600] loss : 0.099
[2,   900] loss : 0.093
[3,   300] loss : 0.079
[3,   600] loss : 0.074
[3,   900] loss : 0.073
[4,   300] loss : 0.063
[4,   600] loss : 0.067
[4,   900] loss : 0.057
[5,   300] loss : 0.052
[5,   600] loss : 0.057
[5,   900] loss : 0.053
[6,   300] loss : 0.047
[6,   600] loss : 0.049
[6,   900] loss : 0.049
[7,   300] loss : 0.044
[7,   600] loss : 0.043
[7,   900] loss : 0.044
[8,   300] loss : 0.037
[8,   600] loss : 0.041
[8,   900] loss : 0.041
[9,   300] loss : 0.036
[9,   600] loss : 0.039
[9,   900] loss : 0.032
[10,   300] loss : 0.033
[10,   600] loss : 0.034
[10,   900] loss : 0.035
accuracy on test set: 98 %

# accuracy并未提高的原因可能是全连接层还是一层
```



### Residual Net

探索：不断将3x3的卷积叠起来，性能未必会随着层数的升高而升高，除了有过拟合问题，还有__梯度消失__的问题

梯度消失：反向传播过程中,由于利用链式法则不断把一堆导数乘起来，假如有小于1的，乘乘乘最后梯度就趋近于0，那权重就得不到更新，网络离输入部分比较近的地方得不到训练。

![CNN09](C:\shelf\Projects\PytorchLeadIn\pics\CNN09.PNG)

Residual net解决梯度消失的问题：加跳连接

正常的话输出的函数就是H(x), Residual在此基础上加了个x

loss对x的导数就因为加的x加上了1，所以回传的东西就都接近了1，就解决了梯度消失

具体实现的时候，右图两个Weight Layer的输出产生的F(x)和x的__张量维度必须一致__

这样的块就叫Residual Block



MNIST分类用简单ResNet结构：

![CNN10](C:\shelf\Projects\PytorchLeadIn\pics\CNN10.PNG)



Residual Block的实现：

![CNN11](C:\shelf\Projects\PytorchLeadIn\pics\CNN11.PNG)

注意：

1.输入输出的Tensor的维度必须一致，这里conv1和conv2的channels都定了一样的，还加了padding保证W和H一致

2.forward里边最后一层先作加法再relu



按照串行关系把Residual Block加进去：

![CNN12](C:\shelf\Projects\PytorchLeadIn\pics\CNN12.PNG)

最后网络就构造完毕了

调试技巧：forward里边一句一句地运行，先把后边注释掉，一点一点给网络添东西；

调试过程中观察网络的输出tensor维度是否符合预期



推荐阅读（各种实现Residual Block的方式的论文）Identity Mappings in Deep Residual Networks

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


def main():
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../datasets/mnist/', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../datasets/mnist/', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.channels = channels
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x):
            y = F.relu(self.conv1(x))
            y = self.conv2(y)
            return F.relu(x+y)

    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

            self.rblock1 = ResidualBlock(16)
            self.rblock2 = ResidualBlock(32)

            self.mp = nn.MaxPool2d(2)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            in_size = x.size(0)

            x = self.mp(F.relu(self.conv1(x)))
            x = self.rblock1(x)
            x = self.mp(F.relu(self.conv2(x)))
            x = self.rblock2(x)

            x = x.view(in_size, -1)
            x = self.fc(x)

            return x

    model = MyNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = criterion(outputs, target)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()
            if batch_idx % 300 == 299:
                print('[%d, %5d] loss : %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    main()


```

【result】

```
[1,   300] loss : 0.490
[1,   600] loss : 0.155
[1,   900] loss : 0.113
[2,   300] loss : 0.086
[2,   600] loss : 0.082
[2,   900] loss : 0.074
[3,   300] loss : 0.067
[3,   600] loss : 0.058
[3,   900] loss : 0.054
[4,   300] loss : 0.050
[4,   600] loss : 0.046
[4,   900] loss : 0.049
[5,   300] loss : 0.044
[5,   600] loss : 0.038
[5,   900] loss : 0.040
[6,   300] loss : 0.034
[6,   600] loss : 0.036
[6,   900] loss : 0.034
[7,   300] loss : 0.031
[7,   600] loss : 0.028
[7,   900] loss : 0.033
[8,   300] loss : 0.028
[8,   600] loss : 0.027
[8,   900] loss : 0.027
[9,   300] loss : 0.023
[9,   600] loss : 0.024
[9,   900] loss : 0.025
[10,   300] loss : 0.023
[10,   600] loss : 0.019
[10,   900] loss : 0.024
accuracy on test set: 99 %

Process finished with exit code 0

```



推荐越多：Densely ..DensNet 跳跳跳

