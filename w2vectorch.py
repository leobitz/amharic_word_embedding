import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *


class Net(nn.Module):

    def __init__(self, embed_size=75,
                 vocab_size=10000,
                 neg_dist=None,
                 neg_samples=5,
                 lr=0.025,
                 device='cpu'):
        super(Net, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.neg_samples = neg_samples
        self.neg_dist = neg_dist
        self.lr = lr
        init_width = 0.5 / embed_size
        x = [np.random.uniform(-init_width, init_width,
                               (vocab_size, embed_size)) for i in range(2)]
        self.device = t.device(device)
        # self.WI = t.tensor(x[0], device=device, requires_grad=True)
        # self.WO = t.tensor(x[0], device=device, requires_grad=True)
        self.WI = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WI.to(device=device, dtype=t.float64)
        self.WI.weight.data.uniform_(-init_width, init_width)
        self.WO = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WO.to(device=device, dtype=t.float64)
        self.WO.weight.data.uniform_(-init_width, init_width)

    def forward(self, x, y):
        x_lookup, y_lookup, neg_lookup = self.prepare_inputs(x, y)

        vI = self.WI(x_lookup)  # .view((-1, ))
        vO = self.WO(y_lookup)  # .view((-1))
        samples = self.WO(neg_lookup)  # .view((-1, 5))

        pos_z = t.mul(vO, vI).squeeze()
        pos_score = t.sum(pos_z, dim=1)
        pos_score = F.logsigmoid(pos_score)
        neg_z = -t.bmm(samples, vI.unsqueeze(2).view(len(x),
                                                     self.embed_size, 1)).squeeze()
        neg_score = F.logsigmoid(neg_z)
        loss = -pos_score - t.sum(neg_score)``
        loss = t.mean(loss)
        return loss

    def prepare_inputs(self, x, y):
        x_lookup = t.tensor([x], dtype=t.long, device=self.device)
        y_lookup = t.tensor([y], dtype=t.long, device=self.device)
        neg_indexes = np.random.randint(
            0, len(self.neg_dist), size=(len(x), self.neg_samples))  # .flatten()
        neg_indexes = self.neg_dist[neg_indexes]  # .reshape((-1, 5)).tolist()
        neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
        return x_lookup, y_lookup, neg_lookup

    def save_embedding(self, word2int, file_name, device):
        if device == 'cpu':
            embedding = self.WI.weight.data.numpy()
        else:
            embedding = self.WI.weight.cpu().data.numpy()
        print(embedding.shape)
        file = open(file_name, encoding='utf8', mode='w')
        file.write("{0} {1}\n".format(len(word2int), self.embed_size))
        for word, index in word2int.items():
            e = embedding[index]
            e = ' '.join(map(lambda x: str(x), e))
            file.write("{0} {1}\n".format(word, e))
        file.close()


def generateSG(data, skip_window, batch_size, start=0, end=-1):
    win_size = skip_window  # np.random.randint(1, skip_window + 1)
    i = win_size
    if end == -1:
        end = len(data)
    while True:
        batch_input = []
        batch_output = []
        for bi in range(0, batch_size, skip_window * 2):
            context = data[i - win_size: i + win_size + 1]
            target = [context.pop(win_size)] * (win_size * 2)
            batch_input.extend(target)
            batch_output.extend(context)
            i += 1
            if i + win_size + 1 > len(data):
                i = win_size

        yield batch_input, batch_output


words = read_file()
words, word2freq = min_count_threshold(words)
# words = subsampling(words, 1e-3)
vocab, word2int, int2word = build_vocab(words)
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))
print("Unk count: ", word2freq['<unk>'])
int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)

n_epoch = 5
batch_size = 5
skip_window = 5
init_lr = .1
gen = generateSG(list(int_words), skip_window, batch_size)

ns_unigrams = np.array(
    ns_sample(word2freq, word2int, int2word, .75), dtype=np.int32)

net = Net(neg_dist=ns_unigrams, vocab_size=len(vocab), device='cpu')
sgd = optimizers.SGD(net.parameters(), lr=init_lr)
start = time.time()
losses = []
grad_time = []
forward_time = []
backward_time = []
step_time = []
steps_per_epoch = (len(int_words) * skip_window) // batch_size
for i in range(steps_per_epoch * n_epoch):
    sgd.zero_grad()
    x, y = next(gen)
    out = net.forward(x, y)
    out.backward()
    sgd.step()
    n_words = i * batch_size
    lr = max(.0001, init_lr * (1.0 - n_words /
                               (len(int_words) * skip_window * n_epoch)))
    for param_group in sgd.param_groups:
        param_group['lr'] = lr
    losses.append(out.detach().cpu().numpy())
    if i % steps_per_epoch == 0:
        print(np.mean(losses), lr)
        losses = []
net.save_embedding(word2int, "results/w2v_plain_torch.txt", 'cpu')


