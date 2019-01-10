import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, max_n_chars, n_consonant, n_vowels, embed_size=100, batch_size=100, lr=0.001):
        super(Net, self).__init__()
        self.max_n_chars = max_n_chars
        self.n_consonant = n_consonant
        self.n_vowels = n_vowels
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.conv_layer = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).cuda().double(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3)
            )
        self.fc1 = nn.Linear(1024, embed_size).cuda().double()
        self.con_rnn = nn.GRU(n_consonant, n_consonant).cuda().double()
        self.vow_rnn = nn.GRU(n_vowels, n_vowels).cuda().double()
        self.con_rnn_hidden = self.init_rnn_hidden()
        self.vow_rnn_hidden = self.init_rnn_hidden()

    def forward(self):
        x = t.randn(100, 1, 13, 50).cuda().double()
        y1 = t.randn(13, 100, 40).cuda().double()
        y2 = t.randn(13, 100, 10).cuda().double()
        x = self.conv_layer(x).view(self.batch_size, -1)
        x = self.fc1(x)
        con_out, self.con_rnn_hidden = self.con_rnn(y1)
        vow_out, self.vow_rnn_hidden = self.vow_rnn(y2)
        print(con_out.shape)
    
    def init_rnn_hidden(self, num=1):
        return t.zeros(1, 1, self.embed_size).cuda().double()

net = Net(13, 40, 10)
net.forward()        
