'''
1min - 10min
真正赋予神经网络能力的是 反向传播自动微分引擎

这节课要做的就是一个  基于自动微分引擎构建的神经网络库

10min
基于一个一元二次方程  3x**2 - 4x + 5
引出了求导公式 f(x + h) - f(x) / h  limit(h-> 0)
可导的条件也是上面的式子存在极限

18min
基于 d = a*b + c
分别用上面的求导公式求了d相对a、b、c的导数

25min左右是画图代码

30min
一个数相对自己本身的导数是1S
30min 30s
叶子节点是神经网络的权重，其他叶子节点或者中间节点是基础数据或者过程数据

我们要计算的是  损失函数相对于权重的导数，因为权重将会利用梯度信息进行迭代更新

34min -- 53min 基础地用公式手动计算相对于loss的导数S
手动利用上面的求导公式算了L相对与前面所有abcdef的导数并且手动填充
S
不过在42min左右引入了链式法则（chain rule）

疑问：
现在学到的都是tensor里面都是参数，然后 data += -learn_rate * grad
第一节课是如何把这个学习率传播下去的？

53min
把输入x比作神经的轴突 axon
把权重比喻成神经的突触 synapse
它俩集合到神经
把偏置比喻成神经自身对输入的反应

还要经过激活，激活就是把 wx+b 左右到激活函数，然它做一个从线性到非线性的变换，可能是压缩函数，比如tanh、ReLU等,得到最终的输出值

55min开始写对应代码

其实就是定义了x1 x2 w1 w2 b
然后各种相加，相乘
注意这里做tanh之前，x1w1 + x2w2 + xnwn + b 这个b是只加一次的

tanh = e**x - e**(-x) / e**x + e**(x)   或者 e**2x + 1 / e**2x - 1

tanh的导数是 （这个有点忘记了，重点记一下） 1 - tanh**2 !! 不太应该忘记，这个和p4的优化有紧密联系，正因为这个公式，才不想让tanh的值太大太小

开始反向传播之前，要赋值loss的grad（也就是开始反向传播的那个数字的grad）

70min 开始代码的反向传播
最开始是一个lambda的空函数，而且在add mul里面还可以继续定义函数，并且使用add mul函数里面的变量

写完了 add mul tan的backward 并且手动调用推导了一次

78min
开始讲拓扑结构，然后准备写反向传播成员函数，完成自动反向传播（其实就是后续遍历）

如果出现了多个同样的Value，那它的grad需要不断累加

a + 2
2 + a
这种都要Value支持

各种加减乘除取反逻辑补充

98min

===下面这个例子可以自己写着试试===

把tanh拆开了 变成e**2x - 1 / e**2x + 1
通过补充的逻辑再反向传播，得到的结果是一样的

100min
还演示了一下pytorch是如何定义变量和反向传播的

104min
开始定义Neuron Layer MLP 来模拟神经网络
这里面的语法要好好学学，简单想了下，虽然知道原理，但是一点都不清晰

自己想的：
1、Neuron
    a、需要知道接受多少个输入，方便创建对应数量的权重w
    b、需要知道输出，才知道对应权重的维度？

    输出是否就是一个值？是否带tanh？

2、Layer
    a、一共有多少个神经元
    b、输出就是所有神经元计算的数组

    注意：还是要关注输入的维度，这决定了layer里面每个neuron的w个数

    对于layer来说，输入的维度决定了这个layer里面所有neuron的w权重个数，输出就是一共需要多少个这样的neuron

3、MLP
    a、入参layer的数组？ xxxxx错误 ===》 是输入 + 输出然后迭代创建layer
    b、x = layer(x) 我记得layer的迭代是这样的 正确
    

    输出是啥？有点抽象。===》就是最后一层layer的数组，只有一维

具体代码见nn.py

112min
引入评价神经网络的方法，loss
均方误差损失，平方的目的是去掉符号，总是正数。loss越大，说明答案越离谱，loss越小越接近0，说明越准确

其他时间都是在敲代码

125min
开始多次backward

131min
作者犯了一个错误，而且是不止一次犯的错误：
在做完一次反向传播以后，不把参数的梯度清零

但是他也说，这就是神经网络比较复杂或者难以发现问题的地方，因为有时候即使少做了一些关键的步骤，依然可以看起来得到比较可信的结论

后面介绍了下他的nn.py和engine.py 还有 test.py

140min
还介绍了他的一个分辨 红蓝点的小李子，和教程略有不同，这个之前没看过

learning rate decay

144min
最后一点，介绍了如果要注册一个新的多项式，应该怎么做

'''


'''
代码学到的
import numpy as np

np.arange(-5, 5, 0.25)


class里面
__init__ 初始化
__repr__ 打印 （representation 的缩写）
__add___ 
__mul___

__init__函数里面，用了空的元组来作为children
_children = ()

e**x 在python里面就是 math.exp(x)
self.backward = lambda: None   不是None 而是空函数


下面是一个topo排序的示例代码, 也是整个backward成员函数的逻辑：
1、topo排序
2、从topo数组反向遍历（其实就是topo排序的反向），也就是从父节点开始做反向传播

topo = []
visited = set()
def build_topo(v):
    if v not in visited: // 先看入参本身在不在，已经加入过就不再加入
        visited.add(v)   
        for child in v._prev:
            build_topo(child) // 把自己的children加入，但是这里会递归，所以是一个深度遍历
        topo.append(v) // 把所有的child都遍历完了才把自己加入，所以这个也是个后序遍历。
build_topo(o)

o.grad = 1
for node in reversed(topo):
    node._backward()

82min

各种反向操作
__radd__
__rmul__
__truediv__
__pow__
__sub__
__neg__

入参判断
other = other if isinstance(other, Value) else Value(other)

e**x 的导数就是它本身 e**x

e**x == x.exp()

100min
micrograd和pytorch的逻辑差不多

import torch
x1 = torch.Tensor([2.0]).double （默认精度是float32）
x1.requires_grad = True
w1/x2/w2/b类似
n = x1*w1 + x2*w2 + b 
o = torch.tanh(n)
o.backward

104min

'''


'''
画图 
import matplotlib.pyplot as plt
plt.plot(xs, ys) --> xs就是f(x) , ys 就是函数的y值

25min
给出了基本的画图代码逻辑（这里就不单独学习了，因为暂时不重要）
'''