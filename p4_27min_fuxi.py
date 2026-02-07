import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 复习内容：
# 1、 最开始的时候，第一轮训练loss太大了，导致前面的训练都在压缩参数，而不是相对有效的训练；所以直接手动设置初始值 最后一层w乘以系数0.01，b设置为0，让神经网络
#     专注于真正的训练，而不是单纯地压缩参数 （0-13min）
# 问题原因：
#   因为W2 b2都是randn初始化，这个函数是高斯分布，在x越来越远离的0的时候，y的值逐渐趋近于1或者-1
#   高斯分布如果“比较矮胖” 比如-3到3的值不算多，那就会导致最后一层有的logits太大或者太小。
#   这样logits如果分布非常不均匀，然后计算loss的时候很可能就因为当前概率太小而导致loss值非常高（也可能loss非常低）
# 解决办法：
#   在最开始的时候，我们认为每个字符串出现的概率都应该是基本平均的，但是例子中最开始是混乱的
#   所以我们可以尽量降低最后一层算logits时的参数值，比如乘以系数0.01，并且也不要偏置的值 （大概在第10分钟）
# 
# 画图：
#   1、画未经过任何优化的lossi，答案是冰球棒 plt.plot(lossi) lossi是一个list
#   2、

# 2、隐藏层的激活值，很多都是1或者-1
# 问题原因：
#    因为tanh是个压缩函数，它会把任意数值都平滑地压缩到-1到1; hpreact是只做了WX + B，h是hpreact.tanh
#    hpreact是一个基本以0中点的正态分布，但是可以看到-5，5两边的数字依然不少，在压缩后，-0.99，0.99之外的数依然很多
#    但是为什么这样会有问题呢？？
#    因为tanh的导数是 (1 - t^2) * parent.grad，如果 t接近1或者-1，那这个参数的梯度就很小了，那它之前的参数的梯度就都很小了，
#    导致前面的神经元的梯度都很小了。 ===> 实际上使用tanh就意味着，梯度在不断地被压缩，因为 (1-t^2) * parent.grad，t又小于等于1
#    有点那个越接近输出层的神经，越能影响ouput。因为层数多了，前面层的神经元的grad就是会小
#    
#    在优化前，我们看到隐藏层里面的数字很多都小于-0.99或者大于0.99
#    但是为什么说，一列，就是32个都是 小于-0.99或者大于0.99 才有问题？
#    因为每个神经元，都要处理32个输入，如果都是 小于-0.99或者大于0.99，那就说明这个神经元无法把任何输入的信息反向传播出去
#    每次学习的时候，基本也无法改变自己的值，因为grad为0，那这个神经元就是 死神经元  （大概到第20分钟）
#    
#    提到了其他的激活函数，sigmoid tanh relu elu都有这个问题。 leaky relu好像不会太严重(0.1x, x)
#解决办法：
#    如何解决初始化的时候很多神经元的tanh值都在1或者-1附近呢？
#    给b1的初始值 *0.01，说是增加一点熵，多一点变化。W1 * 0.2
#    这样hpreact就在0附近更集中了，h（hpreact.tanh()）也就好了很多，-1和1附近的数明显减少
#其他：
#    神经元在训练的过程中可能被“杀死”

words = open("names.txt", 'r').read().splitlines()

listchar = list(set(''.join(words)))
listchar = sorted(listchar)

itos = { i + 1 : c for i, c in enumerate(listchar) }
itos[0] = '.'
stoi = { i:s for s, i in itos.items()}
vocab_size = len(itos)
print(itos)
print(stoi)
print(vocab_size)

block_size = 3

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xtdev, Ydev = build_dataset((words[n1:n2]))
xte, Yte = build_dataset(words[n2:])

# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd),             generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    # 就是从训练集里面 随机选batch_size(最开始赋值是32)个前缀行
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    # 又是这种广播赋值  Xtr[10000, 3]  ix[32] = [10,77, .. ,99] 随机32个
    # 选出ix里面个数的Xtr形成Xb 那Xb就是 [32, 3]
    Xb, Yb = Xtr[ix], Ytr[ix]   

    # forward pass
    # a[b]：返回形状为 b.shape + a.shape[1:] 的数组
    # C[27, 10]， Xb[32, 3] 32 * 3每一个都形成一个32 * 3 * 10 的数字 
    emb = C[Xb] # emb [32, 3, 10]
    # -1 的意思就是把后面所有维度都展开
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    h = hpreact.tanh()
    logits = h @ W2 + b2
    # softmax:
    # logcount = logits.exp()
    # 这里理解的还是不熟练，勉强推导出来了。
    # prob = logcount / logcount.sum(1, keepdim = True)
    # 通过logits和yb（答案）就可以softmax并且通过对比答案计算loss
    loss = F.cross_entropy(logits, Yb)                                                                                                              

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 10000 else 0.01 # 学习率在后面渐渐下降
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())
    
    # 第二个问题 隐藏层的激活值很多-1或者1
    # print(h)
    # plt.hist(hpreact.view(-1).tolist(), 50); # histogram 直方图
    # plt.hist(h.view(-1).tolist(), 50); # histogram 直方图
    # plt.show()
    # 第二个问题 h的值大于0.99有多少
    # plt.figure(figsize=(20,10))
    # plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
    # plt.show()
    # break

# 第二个问题 冰球棒形状
# plt.plot(lossi)
# plt.show()

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val'  : (Xtdev, Ydev),
        'test' : (xte, Yte)
    } [split]

    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

# 推理
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size
    # print(torch.tensor([context]).shape)
    # 开始从0推理
    while True:
        # context 是[1, 3], C是[27, 10], 所以根据上面规则就是context.shape（1， 3） + C[1:]（10） ==> (1, 3, 10)
        # 注意，这里是1，3 不是 3
        emb = C[torch.tensor([context])]
        # print("tuili emb shape", emb.shape)
        h = emb.view(1, -1) @ W1 + b1
        h = h.tanh()
        logits = h @ W2 + b2
        # dim = 1是什么意思？
        probs = F.softmax(logits, dim=1)
        # 采样
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break;

    print(''.join(itos[i] for i in out))



# 15:15 -- 16:20
# 16:20 -- 16:45 休息
# 16:45 -- 17:00 
# 17:00 -- 17:11 休息
# 17:11 -- 17:26 看了会儿采访
'''
1. 
'''