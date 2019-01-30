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

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=5,
                      stride=1, padding=2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer2 = nn.Linear(640, embed_size).double()
        self.layer3 = nn.Linear(embed_size, embed_size).double()
        self.T = nn.Sequential(
            nn.Linear(embed_size, embed_size).double(),
            nn.Sigmoid()
        )
        self.TX = nn.Parameter(t.rand(embed_size), requires_grad=True).double()


    def vI_out(self, x_lookup, word_image, batch_size):
        # x = self.layer1(word_image)
        # x = x.view(batch_size, -1)
        # x = self.layer2(x)
        # y = self.layer3(x)
        y =  self.WI(x_lookup)
        # T = self.T(y)
        # T = F.sigmoid(self.TX)
        # C = 1 - T
        # z = y * T + x * C
        return [y]

    def forward(self, word_image, x, y):
        word_image, x_lookup, y_lookup, neg_lookup = self.prepare_inputs(
            word_image, x, y)

        vO = self.WO(y_lookup)
        samples = self.WO(neg_lookup)
        out = self.vI_out(x_lookup, word_image, len(y))
        vI = out[0]

        pos_z = t.mul(vO, vI).squeeze() 
        vI = vI.unsqueeze(2).view(len(x), self.embed_size, 1)
        neg_z = -t.bmm(samples, vI).squeeze()

        pos_score = t.sum(pos_z, dim=1)
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(neg_z)

        loss = -pos_score - t.sum(neg_score)
        loss = t.mean(loss)
        return loss

    def prepare_inputs(self, image, x, y):
        word_image = t.tensor(image, dtype=t.double, device=self.device)
        y_lookup = t.tensor(y, dtype=t.long, device=self.device)
        x_lookup = t.tensor(x, dtype=t.long, device=self.device)
        neg_indexes = np.random.randint(
            0, len(self.neg_dist), size=(len(y), self.neg_samples))  # .flatten()
        neg_indexes = self.neg_dist[neg_indexes]  # .reshape((-1, 5)).tolist()
        neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
        return word_image, x_lookup, y_lookup, neg_lookup

    def get_embedding(self, image, x):
        word_image = t.tensor(image, dtype=t.double, device=self.device)
        x_lookup = t.tensor(x, dtype=t.long, device=self.device)
        out = self.vI_out(x_lookup, word_image, len(x))
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
