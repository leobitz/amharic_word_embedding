import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *


class Net(nn.Module):

    def __init__(self, embed_size=100,
                 vocab_size=10000,
                 neg_dist=None,
                 neg_samples=5,
                 lr=0.025,
                 seq_encoding=None,
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
        if seq_encoding is not None:
            self.seq_embed = nn.Embedding(
                seq_encoding.shape[0], seq_encoding.shape[1])
            self.seq_embed.to(device=device, dtype=t.float64)
            self.seq_embed.weight.data.copy_(
                t.from_numpy(seq_encoding))
            self.seq_embed.weight.requires_grad = False

        self.WI = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WI.to(device=device, dtype=t.float64)
        self.WI.weight.data.uniform_(-init_width, init_width)
        self.WO = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WO.to(device=device, dtype=t.float64)
        self.WO.weight.data.uniform_(-init_width, init_width)
        if device == 'cuda':
            self.fc1 = nn.Linear(
                seq_embed.shape[1], seq_embed.shape[1]).cuda().double()
            self.fc2 = nn.Linear(
                (embed_size + seq_embed.shape[1]), embed_size).cuda().double()
        else:
            self.fc1 = nn.Linear(
                seq_embed.shape[1], seq_embed.shape[1]).double()
            self.fc2 = nn.Linear(
                (embed_size + seq_embed.shape[1]), embed_size).double()

    def forward(self, x, y):
        x_lookup, y_lookup, neg_lookup = self.prepare_inputs(x, y)

        vI = self.WI(x_lookup)
        vO = self.WO(y_lookup)
        samples = self.WO(neg_lookup)

        # seqI = self.seq_embed(x_lookup)
        seqO = self.seq_embed(y_lookup)
        neg_seq = self.seq_embed(neg_lookup)

        samples = self.fc1(samples)
        # seqI = self.fc1(seqI)
        seqO = self.fc1(seqO)

        samples = t.cat((samples, neg_seq), 2)
        vO = t.cat((vO, seqO), 2)
        # vI = t.cat((vI, seqI), 2)
        # vI = self.fc2(vI)
        vO = self.fc2(vO)
        samples = self.fc2(samples)

        pos_z = t.mul(vO, vI).squeeze()
        pos_score = t.sum(pos_z, dim=1)
        pos_score = F.logsigmoid(pos_score)
        neg_z = -t.bmm(samples, vI.unsqueeze(2).view(len(x),
                                                     self.embed_size, 1)).squeeze()
        neg_score = F.logsigmoid(neg_z)
        loss = -pos_score - t.sum(neg_score)
        loss = t.mean(loss)
        return loss

    def get_embedding(self, x):
        x_lookup = t.tensor([x], dtype=t.long, device=self.device)
        vI = self.WI(x_lookup)
        seqI = self.seq_embed(x_lookup)
        seqI = self.fc1(seqI)
        vI = t.cat((vI, seqI), 2)
        vI = self.fc2(vI)
        return vI.detach().numpy().reshape((-1, self.embed_size))

    def prepare_inputs(self, x, y):
        x_lookup = t.tensor([x], dtype=t.long, device=self.device)
        y_lookup = t.tensor([y], dtype=t.long, device=self.device)
        neg_indexes = np.random.randint(
            0, len(self.neg_dist), size=(len(x), self.neg_samples))
        neg_indexes = self.neg_dist[neg_indexes]
        neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
        return x_lookup, y_lookup, neg_lookup

    def save_embedding(self, word2int, file_name, device):
        file = open(file_name + '_', encoding='utf8', mode='w')
        file.write("{0} {1}\n".format(len(word2int), self.embed_size))
        with t.no_grad():
            xs = list(word2int.values())
            es = self.get_embedding(xs)
            for word, index in word2int.items():
                e = es[index]
                e = ' '.join(map(lambda x: str(x), e))
                file.write("{0} {1}\n".format(word, e))
        file.close()

        if device == 'cpu':
            embedding = self.WI.weight.data.numpy()
        else:
            embedding = self.WI.weight.cpu().data.numpy()

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


def get_new_embedding(oldw2i, neww2i, embeddings):
    new_em = np.ndarray((len(neww2i), embeddings.shape[1]), dtype=np.float32)
    for key in neww2i:
        if key != "<unk>":
            index = oldw2i[key]
            em = embeddings[index]
            new_em[neww2i[key]] = em
    new_em[0] = np.random.rand(100)
    return new_em


seq_embed = np.load('results/seq_best.npy')

words = read_file()
xvocab, xword2int, xint2word = build_vocab(words)
words, word2freq = min_count_threshold(words)

vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)

seq_embed = get_new_embedding(xword2int, word2int, seq_embed)
seq_embed = normalize(seq_embed)
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))
print("Unk count: ", word2freq['<unk>'])
int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)

n_epoch = 2
batch_size = 10
skip_window = 1
init_lr = .1
device = 'cpu'
gen = generateSG(list(int_words), skip_window, batch_size)

ns_unigrams = np.array(
    ns_sample(word2freq, word2int, int2word, .75), dtype=np.int32)

net = Net(neg_dist=ns_unigrams, seq_encoding=seq_embed,
          vocab_size=len(vocab), device=device)
sgd = optimizers.SGD(net.parameters(), lr=init_lr)
start = time.time()
losses = []
grad_time = []
forward_time = []
backward_time = []
step_time = []
window = skip_window * 2
steps_per_epoch = (len(int_words) * window) // batch_size
start_time = time.time()
total_steps = steps_per_epoch * n_epoch
for i in range(total_steps):
    sgd.zero_grad()
    x, y = next(gen)
    out = net.forward(x, y)
    out.backward()
    sgd.step()
    n_words = i * batch_size
    lr = max(.0001, init_lr * (1.0 - n_words /
                               (len(int_words) * window * n_epoch)))
    for param_group in sgd.param_groups:
        param_group['lr'] = lr
    losses.append(out.detach().cpu().numpy())
    if i % (steps_per_epoch // 10) == 0:
        s = "Loss: {0:.4f} lr: {0:.4f} Time Left: {2:.2f}"
        span = (time.time() - start_time)
        print(s.format(np.mean(losses), lr, span))
        start_time = time.time()
        losses = []
net.save_embedding(word2int, "results/w2v_torch.txt", device)
