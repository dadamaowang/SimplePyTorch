# 08 Softmax Classifier

以MNIST手写数据集分类为例：

对输入的图片，分出的类别有数字0~9这10个；

那么，对于10个目标类，输出的结果要有一个__相互抑制__的作用；

但是如果仅仅是输出属于某个类别的概率是多少，上述抑制作用是体现不出来的；

比如对y1可能性是0.8,y2的可能性也是0.8,y3是0.9...那样子并不能判断到底是哪个数字；

所以要求输出满足一种__分布性质__的要求：

- all outputs > 0
- $\sum outputs =1$

这样才能真正算出样本属于各个分类的__分布__

the outputs is competetive!

__distribution__



搭建网络结构：

![s01](C:\shelf\Projects\PytorchLeadIn\pics\s01.PNG)

最后一层需要用到softmax

## Softmax

suppose $Z^l \in R^K$is the output of the last linear layer, the Softmax function:
$$
P(y = i) = \frac{e^{z^i}}{\sum_{j=0}^{K-1}e^{z^j}} , i \in \{0, 1, ...K-1\}
$$
假设一共K个分类，先计算它的指数（保证大于0）；

分母就是每一个分类的输出的求和（保证结果的和正好是1）



在Pytorch里计算交叉熵损失：

就是下图的softmax,loss啥啥的合并到一起去，算是CrossEntropyLoss

![s02](C:\shelf\Projects\PytorchLeadIn\pics\s02.PNG)

使用的时候注意y要是LongTensor类型的

使用实例：pred给出的list是分别代表了三个类别属于哪一类的概率，显然pred1的预测更好一些；

CrossEntropyLoss把来自pred的全连接层的输出弄成经过softmax之后的一个元组，再计算损失的值，最后得出一个损失值的标量

![s03](C:\shelf\Projects\PytorchLeadIn\pics\s03.PNG)





## MNIST 识别例子

图像：每个像素映射到0-255的一个矩阵

模型简介：

![s04](C:\shelf\Projects\PytorchLeadIn\pics\s04.PNG)

Relu作激活层；

最后一层output不作激活，由交叉熵损失CrossEntropyLoss来作softmax激活



【code:】

```python
import torch
from torchvision import transforms  # 帮助图像处理的
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as ff
import torch.optim as optim


def main():
    batch_size = 64
    transform = transforms.Compose([  # Convert PIL Image to piexl Tensor[0,1]
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化：前边是均值，后边是标准差
        # 希望输入的数据能够满足0,1间的标准分布
        # 这里俩数字是算的，别的数据集也要计算一下
        # pix_norm = (pix_orig - mean) / std
    ])
    # channel数目是不一样的
    # 灰度图是单通道的
    # 读进来的时候一般都是 W * H * C (宽高通)
    # Pytorch 里边都先转换成 C * W * H ,便于高效处理
    # 所以要先变成一个torch里的tensor,可以用ToTensor实现
    # Compose可以将transform里边这些作一个pipline处理

    # mnist.py的URL已更改为本地
    train_dataset = datasets.MNIST(root='../datasets/mnist/', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../dataset/mnist/', train=True,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    class MnistClassifier(torch.nn.Module):
        def __init__(self):
            super(MnistClassifier, self).__init__()  # input: 1*28*28三阶张量变成1阶
            self.l1 = torch.nn.Linear(784, 512)  # 降维
            self.l2 = torch.nn.Linear(512, 256)  # 全连接网络只要保证参数对应相等就能连接起来
            self.l3 = torch.nn.Linear(256, 128)
            self.l4 = torch.nn.Linear(128, 64)
            self.l5 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 784)  # 第一个数-1表示自动根据输入的大小计算
            # view展平变成二阶矩阵，得到N*784的矩阵
            x = ff.relu(self.l1(x))  # 用relu激活一波
            x = ff.relu(self.l2(x))
            x = ff.relu(self.l3(x))
            x = ff.relu(self.l4(x))
            return self.l5(x)  # 注意最后一层不作激活，直接把它放入自定义的softmax里

    model = MnistClassifier()
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 带冲量优化训练过程

    for epoch in range(10):
        print('epoch:', epoch)
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_val = criterion(outputs, target)
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()

            if batch_idx % 300 == 299:  # 每个epoch中iteration为300的时候打印一次，不然太多了
                print('[epoch, batch_idx][%d, %5d], loss : %3f'
                      % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0

    # test过程
    correct = 0  # 正确预测的数目
    total = 0  # 总数
    with torch.no_grad():  # test不需要计算梯度的，所以用with后边不要梯度，
        # 这样这部分代码就都不会计算梯度，省资源提高效率
        for data in test_loader:
            images, labels = data
            output = model(images)
            # 这里output就是一个批次的样本的一个结果矩阵：
            # 比如这个batch里边有batch_size个样本
            # 那矩阵的行数就是batch_size
            # 每一行一个tensor有10列的量，也就是网络里边没作交叉熵损失的那个output的LongTensor
            # 从中找哪一列的值最大，就最有可能是哪个，然后它的下标就是结果了
            _, predicted = torch.max(output.data, dim=1)
            # 这里使用torch的max函数，沿着维度dim=1（也就是结果矩阵的每个行）输出最大的值
            # 返回的俩东西，"_"是那个最大值（并不需要），predicted是最大值位置的下标
            total += labels.size(0)
            # 先把总数加上。这里label就是个N*1(batch_size*1)的矩阵，
            # 所以size就是(N,1)这个元组，取第0个元素
            correct += (predicted == labels).sum().item()
            # 然后predicted和label作比较，真就是1，假就是0，求个和把标量拿出来就行
    print('Accuracy on the test: %d, %%' % (100 * correct / total))  # 张量之间的比较预算


if __name__ == '__main__':
    main()

```



【result:】

```
epoch: 0
[epoch, batch_idx][1,   300], loss : 2.168950
[epoch, batch_idx][1,   600], loss : 0.846045
[epoch, batch_idx][1,   900], loss : 0.411627
epoch: 1
[epoch, batch_idx][2,   300], loss : 0.317232
[epoch, batch_idx][2,   600], loss : 0.265585
[epoch, batch_idx][2,   900], loss : 0.231924
epoch: 2
[epoch, batch_idx][3,   300], loss : 0.183910
[epoch, batch_idx][3,   600], loss : 0.169954
[epoch, batch_idx][3,   900], loss : 0.159757
epoch: 3
[epoch, batch_idx][4,   300], loss : 0.132731
[epoch, batch_idx][4,   600], loss : 0.124928
[epoch, batch_idx][4,   900], loss : 0.117259
epoch: 4
[epoch, batch_idx][5,   300], loss : 0.098618
[epoch, batch_idx][5,   600], loss : 0.094962
[epoch, batch_idx][5,   900], loss : 0.096985
epoch: 5
[epoch, batch_idx][6,   300], loss : 0.079873
[epoch, batch_idx][6,   600], loss : 0.074706
[epoch, batch_idx][6,   900], loss : 0.075688
epoch: 6
[epoch, batch_idx][7,   300], loss : 0.058655
[epoch, batch_idx][7,   600], loss : 0.062287
[epoch, batch_idx][7,   900], loss : 0.063220
epoch: 7
[epoch, batch_idx][8,   300], loss : 0.051736
[epoch, batch_idx][8,   600], loss : 0.051406
[epoch, batch_idx][8,   900], loss : 0.048625
epoch: 8
[epoch, batch_idx][9,   300], loss : 0.038459
[epoch, batch_idx][9,   600], loss : 0.042441
[epoch, batch_idx][9,   900], loss : 0.045847
epoch: 9
[epoch, batch_idx][10,   300], loss : 0.029880
[epoch, batch_idx][10,   600], loss : 0.034237
[epoch, batch_idx][10,   900], loss : 0.038015
Accuracy on the test: 99, %
```



小结：如果加上特征提取效果会更好（傅里叶，小波等等）,但是都是人工的；

CNN可以用于自动特征提取

