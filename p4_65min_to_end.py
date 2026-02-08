# 65min
"""
开始讲了一下ResNet，用于图像分类的神经网络
里面的bottleneck类就有很多我们现在学的，比如参数初始化，forward，backward等等

forward里面也包含了卷积层，因为是用于图像的（就是我们用的线性层），批量归一化层，非线性层ReLU（我们用的tanh）

值得注意的是，里面有个Conv1x1接口，它调用nn.Conv2d()的时候，有个参数是bias=False, 这个的作用就相当于是之前我们干掉了b1一样

"""
# 69min
"""
开始讲pytorch里面的一些类
1、 torch.nn.Linear
torch.nn.Linear(in_features,out_features, bias=True, device=None, dtype=None)
Linear.weight 默认初始化是 u(-sqrt(k), sqrt(k)), k = 1 / in_feature, u代表均匀分布  
    ==> 这个和代码 * 5 / 3 / (n_embedding * size) ** 0.5 基本一样，只不过他们gain是1 不是 5 / 3
Linear.bias 一样的 u(-sqrt(k), sqrt(k)), k = 1 / in_feature

2、torch.nn.BatchNorm1d(num_feature, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
num_feature 就是最终Linear层的最后一维数量
eps 公式防止除0，默认 1e-05
momentum 就是为了那个 mean/std running  和 meani stdi 的变化值，相当于之前我们设置的是0.001
简单来说，如果batch size比较小，就建议更小点
affine 决定这个批量归一化层是否有学习的参数 bngain bnbias，找不到理由不设置为false

"""
# 74min - 79min 在总结这节课的内容
# 79min - 最后  把之前零散的代码总结成和 torch.nn相似的代码，这样更容易理解，并且做了些图让大家更好理解参数的辩护