# 27min开始到65min
import torch
'''
torch.randn初始化的数字，要如何改变才能让 x @ w它的值变成标准差为1 --- 要除以 fan_in ** 0.5，fan_in 两个矩阵相连的数字
比如x = torch.randn(1000, 10) w = torch.randn(10,100), y = x @ w, 那就要除以  10**0.5

论文细节《Diving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》
论文中提到了ReLU和PReLU，提到了如何精确地调整输入
backward也有提及，但是因为是个敞亮，不是很重要。

torch.nn.init.kaiming_normal_ 这个kaiming提的方法已经进入了标准库。mode一般就默认fan_in，影响不大
nonlinearity 非线性函数 比如 Leaky_relu / relu / tanh / sigmoid ...

gain值的计算
linear / identity / sigmoid / Conv{1,2,3}D (这是什么 --- 卷积层)== 1
tanh = 5/3
ReLU = 2**0.5
Leaky Relu (2 / (1 + 斜率**2))**0.5

为什么需要gain（增益）呢？
因为在计算隐藏层的时候，tanh就像ReLU一样，是一个压缩变换，它会把头尾的数据压缩在很小的空间内，所以我们要对抗一下这个压缩的动作，并且把
内容从新归一化回到单位标准差

但是这个对于激活值的梯度范围的初始化设置的文章是7年前提出的，在当时你必须很小心地去设置它，但是现在有更创新的方法，比如残差链接，多种归一化层
，更多优化器，Adam等。所以这些初始化的设置变得不是那么重要。
不过作者还是对w1 进行了  torch.randn(10, 1000) * 5 / 3 / fann**0.5的操作

40min 开始讲批量归一化层
最开始人们希望hpreact的值能够绝对值不要太大或者太小，而是呈高斯分布，这样子的话后面tanh的时候不至于太大或者太小
因为
hpreact绝对值太小，求梯度的时候会导致梯度为1，但是激活值太小这样就传递不了信息。
hpreact绝对值太大，梯度会下降到0，导致反向传播的时候权重很难更新。

'''

test_p = torch.randn(10000).std()
print(test_p)


# 54min
# 我们希望在训练好一个模型后，能在环境中部署它，并且能够输入单个样本并且从我们的神经网络中得到结果。
# 但是根据之前的优化，forward的时候，求logits是要根据前面一批的输入才能被计算出来计算，这就和推理时的单个样本冲突
# 论文建议：在训练后，计算并且设置一次训练集上的批量归一化均值和标准差。 这个过程称之为： 校准批量归一化统计数据
# 方法1：
# 可以用with 语句来对一小段代码进行装饰
# 其实就是用整个训练集做embedding作为输入，hpreact = X @ W1 + b1
# 然后计算全量训练集的hpreact的结果的平均值和标准差
# 在推理的时候，计算loss时，就用这个固定的额值：hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
# 
# 方法2
# 每个阶段以0.001的临时量来更新，这样就不用最后算一次了
#with torch.no_grad()
# bnmean = 0.999 * bnmean + 0.001 * bnmeani
# bnstd = 0.999 * bnstd + 0.001 * bnstdi
# 
# 1:01:00  解释归一化时的epsilon, 它默认很小 10^(-5),防止公式除以0
# X @ W1 + b1 ， bias在计算的时候实际上是被减掉了，所以没用
# 因为 hpreact = embcat @ W1 + b1, bnmeani = hpreact.mean(0, keepdim=True)
# 然后重新计算hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias ， (hpreact - bnmeani) 这个实际上是把b1又减去了
# b1这种偏移的功能现在已经被 bnbias替代了
# 1:03:00 总结：
"""
关于批量归一化层 BatchNormLayer 的简要总结：

我们使用批量归一化来控制神经网络中激活值的统计特性，在神经网络中广泛使用批量归一化层带来控制激活值的统计特性；通常，我们会在
包含乘法运算层（线性层或者卷积层）之后放置批量归一化层。批量归一化的内部有用于增益和偏差的参数，这些参数是通过反向传播
训练得到的。
它还有两个缓冲区，分别是运行时的均值和标准差。这些并不是反向传播训练而来，而是向“运行时的均值更新”来训练的

批量归一化层的步骤：
计算输入到激活值在该批次上的均值和标准差，并将该批次数据中心化，使其成为单位高斯分布，然后它通过学习到的偏差和增益对其进行偏移（bias）和缩放（gain）。
在此基础上， 批量归一化层还记录了输入的均值和标准差，并且它保持着这个运行中的均值和标准差。这会用在后续的推理过程中，而不用重新估计均值。

1:05:00
一个实际的例子：ResNet，它是个残差神经网络，提到了 bottleneck block。
算了，这个明天再看
"""
