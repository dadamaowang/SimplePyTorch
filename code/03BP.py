# 简单反向传播示例
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # 记得中括号
w.requires_grad = True  # 设置为True表示对变量w自动求梯度
# 默认的Tensor不会计算关于w的梯度，这个要自己设定


# 模型
def forward(x):  # x是个tensor了，这里是tensor和tensor之间的乘法,得到的还是tensor
    return w * x
    # *运算符已经被重载了，要进行tensor与tensor之间的数乘
    # x被自动类型转换为tensor
    # 计算出节点的输入值也需要包含梯度


def loss(x, y):  # 损失
    y_pred = forward(x)
    return (y_pred - y) ** 2
# 看到这些运算就反应过来是构建计算图
# 从能画出计算图的角度考虑代码


for epoch in range(100):  # 训练100轮
    print('epoch: ', epoch)
    for x, y in zip(x_data, y_data):
        loss_value = loss(x, y)  # 前馈
        loss_value.backward()
        # 自动把计算图上的计算链路上所有要梯度的都计算出来
        # backward完毕后，把梯度存在需要的地方（变量自己里边）
        # 只要作完backward，这个计算图就没有了
        # 下次计算是新的计算图
        # 每进行一次反向传播，释放一次计算图，是一种灵活的方式
        print('\tgrad : ', w.grad.item())
        # w.grad获得梯度；
        # 张量里边的data和grad中，grad也是tensor
        # .item()为了只把数值拿出来当作标量

        # 所以要取出来data进行计算，（取data不会建立计算图）
        w.data = w.data - 0.01 * w.grad.data  # 因为这里只想作纯数值修改

        w.grad.data.zero_()
        # 一定显式清0，否则w中每次计算出的梯度会累加
        # 本例中由于就想一个样本一个样本地训，就不要累加
        # 事实上还有很多技巧是需要累加的，这就是w会自动累加的原因
        # 在不用那些技巧的时候，手动清零很必要

    print('loss_val:', loss_value.item())
    # notice: 打印一定是.item()取出来里边的标量









