import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g)
W2 = torch.randn((n_hidden, vocab_size),          generator=g)
b2 = torch.randn(vocab_size,                      generator=g)

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 20000
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
