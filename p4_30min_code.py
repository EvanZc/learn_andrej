import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1、优化了最开始第一次训练后loss太高的问题
# 2、优化了“死亡细胞的问题”

# 20260131 19：53 - 21:04:23

words = open("names.txt", 'r').read().splitlines()

listchar = list(set(''.join(words)))

# 沃日啊，这里是返回排序以后的listchar
listchar = sorted(listchar)

itos = { i + 1 : c for i, c in enumerate(listchar) }
itos[0] = '.'

# 自己的方法1  因为listchar依然没有 . 所以还是要补充一个
stoi = { c : i + 1 for i, c in enumerate(listchar) }
stoi['.'] = 0
# 方法2，这里可以直接遍历set，并且把set的index作为value 
# itos = { i:s for s, i in stoi.items()}

print(itos)
print(stoi)


# 这个random没有想到
import random
random.seed(42)
# random.shuffle(words)

# 构造训练集
X, Y = [], []
block_size = 3
for w in words[:5]:
    context = [0] *  block_size
    for ch in w + '.':
        X.append(context)
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]

# 这一段代码可以优化
len_x1 = int(len(X) * 0.8)
len_x2 = int(len(X) * 0.9)

len_y1 = int(len(Y) * 0.8)
len_y2 = int(len(Y) * 0.9)

# 这里不用区分是不是在句号分开
Xtr = X[:len_x1]
Xdev = X[len_x1: len_x2]
Xte = X[len_x2 :]

Ytr = Y[:len_y1]
Ydev = Y[len_y1: len_y2]
Yte = Y[len_y2 :]

# 最开始没有想到要把list要转换成tensor
X = torch.tensor(X)
Y = torch.tensor(Y)
Xtr = torch.tensor(Xtr)
Xdev = torch.tensor(Xdev)
Xte = torch.tensor(Xte)

Ytr = torch.tensor(Ytr)
Ydev = torch.tensor(Ydev)
Yte = torch.tensor(Yte)

print(X) 
print(Y)

# 训练前构造参数
char_num = 27
embedding_dim = 2
# batch后面才有用？
input_batch = 32
layer_1_neuron = 100
train_num = 50
learning_rate = 0.1

# embedding table
C = torch.randn((char_num, embedding_dim))
# 第一层神经网络
W1 = torch.randn((block_size * embedding_dim, layer_1_neuron))
b1 = torch.randn((layer_1_neuron))
# 最后一层神经网络
W2 = torch.randn((layer_1_neuron, char_num))
b2 = torch.randn((char_num))
# 参数表
params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad = True
# 训练
def train(train_num, input_X, ans_Y):
    for _ in range(train_num):
        # 先通过embedding表查找C
        input = C[input_X]
        input = input.view(input_X.shape[0], block_size * embedding_dim) 
        # 第一层计算 WX + b ，再tanh
        res = (input @ W1 + b1).tanh()
        # 第二层计算, 不用tanh
        logits = (res @ W2 + b2)
        # soft_max
        logcount = logits.exp()
        prob = logcount / logcount.sum(1, keepdim=True)
        # print("prob shape is ", prob.shape)
        # loss计算 --- 这个广播还是没太理解
        loss = -prob[torch.arange(input_X.shape[0]), ans_Y].log().mean()
        # 反向传播
        for p in params:
            p.grad = None
        loss.backward()
        print(loss.item())
        # 学习
        for p in params:
            p.data -= learning_rate * p.grad

train(train_num, Xtr, Ytr)

'''
1. 
'''