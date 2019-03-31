import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, embed_size=75,
                 vocab_size=10000,
                 neg_dist=None,
                 neg_samples=5,
                 lr=0.025,
                 device='cpu', 
                 seq_encoding=None):
        super(Net, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.neg_samples = neg_samples
        self.neg_dist = neg_dist
        self.lr = lr

        init_width = 0.5 / embed_size
        x = [np.random.uniform(-init_width, init_width,
                               (vocab_size, embed_size)) for i in range(2)]
        if seq_encoding is not None:
            self.seq_embed = nn.Embedding(
                seq_encoding.shape[0], seq_encoding.shape[1])
            self.seq_embed.to(device=device, dtype=t.float64)
            self.seq_embed.weight.data.copy_(
                t.from_numpy(seq_encoding))
            self.seq_embed.weight.requires_grad = False

        if device == 'gpu':
            device = 'cuda'
        self.device = t.device(device)

        self.WI = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WI.to(device=device, dtype=t.float64)
        self.WI.weight.data.uniform_(-init_width, init_width)
        self.WO = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WO.to(device=device, dtype=t.float64)
        self.WO.weight.data.uniform_(-init_width, init_width)

        n_filters = 10

        self.fc1 = nn.Linear(200, 100).double()
        self.fc2 = nn.Linear(200, 100).double()
        self.T = nn.Parameter(t.tensor(np.random.rand(embed_size), requires_grad=True, device=device, dtype=t.float64))

    def vI_out(self, x_lookup, y_lookup, batch_size):
        y =  self.WI(x_lookup)
        seqI = self.seq_embed(x_lookup)
        seqO = self.seq_embed(y_lookup)
        # seqI = seqI * self.T
        temp = t.cat((seqO, seqI), 1)
        temp = self.fc1(temp)
        vi = self.fc2(t.cat((temp, y), 1))
        # vi = y + temp

        return [vi, y, temp]
    
    def vI_out2(self, x_lookup, batch_size):
        y =  self.WI(x_lookup)
        # seqI = self.seq_embed(x_lookup)
        # seqO = self.seq_embed(y_lookup)
        # seqI = seqI * self.T
        # temp = t.cat((seqO, seqI), 1)
        # temp = self.fc1(temp)
        # vi = y + seqI

        return [y]

    def forward(self, x, y):
        x_lookup, y_lookup, neg_lookup = self.prepare_inputs( x, y)

        vO = self.WO(y_lookup)
        vI = self.WI(x_lookup)

        seqO = self.seq_embed(neg_lookup)
        seqO = seqO * self.T
        samples = self.WO(neg_lookup)
        samples = seqO + samples
        # vI = self.seq_embed(x_lookup) * self.T + vI

        vO = vO + self.T * self.seq_embed(y_lookup)
        # out = self.vI_out(x_lookup, len(y))
        # vI = out[0]

        pos_z = t.mul(vO, vI).squeeze() 
        vI = vI.unsqueeze(2).view(len(x), self.embed_size, 1)
        neg_z = -t.bmm(samples, vI).squeeze()

        pos_score = t.sum(pos_z, dim=1)
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(neg_z)

        loss = -pos_score - t.sum(neg_score)
        loss = t.mean(loss)
        return loss

    def prepare_inputs(self,  x, y):
        # word_image = t.tensor(image, dtype=t.double, device=self.device)
        y_lookup = t.tensor(y, dtype=t.long, device=self.device)
        x_lookup = t.tensor(x, dtype=t.long, device=self.device)
        neg_indexes = np.random.randint(
            0, len(self.neg_dist), size=(len(y), self.neg_samples))  # .flatten()
        neg_indexes = self.neg_dist[neg_indexes]  # .reshape((-1, 5)).tolist()
        neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
        return x_lookup, y_lookup, neg_lookup

    def get_embedding(self, image, x):
        # word_image = t.tensor(image, dtype=t.double, device=self.device)
        x_lookup = t.tensor(x, dtype=t.long, device=self.device)
        out = self.vI_out2(x_lookup, len(x))
        result = [r.detach().numpy() for r in out]
        return result

    def save_embedding(self, embed_dict, file_name, device):
        file = open(file_name, encoding='utf8', mode='w')
        file.write("{0} {1}\n".format(len(embed_dict), self.embed_size))
        for word in embed_dict.keys():
            e = embed_dict[word]
            e = ' '.join([str(x) for x in e])
            file.write("{0} {1}\n".format(word, e))
        file.close()
