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
    print('Accuracy on the test: %d %%' % (100 * correct / total))  # 张量之间的比较预算


if __name__ == '__main__':
    main()




