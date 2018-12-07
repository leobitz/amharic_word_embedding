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
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.fc1 = nn.Linear(16 * 4 * 16, embed_size).double()
        self.WO = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.WO.to(device=device, dtype=t.float64)
        self.WO.weight.data.uniform_(-init_width, init_width)

    def forward(self, word_image, x, y):
        word_image, y_lookup, neg_lookup = self.prepare_inputs(word_image, y)
        input_x = self.layer1(word_image).view(len(y), -1)

        vI = self.fc1(input_x)
        vO = self.WO(y_lookup)
        samples = self.WO(neg_lookup)

        pos_z = t.mul(vO, vI).squeeze()
        vI = vI.unsqueeze(2).view(len(x), self.embed_size, 1)
        neg_z = -t.bmm(samples, vI).squeeze()

        pos_score = t.sum(pos_z, dim=1)
        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(neg_z)

        loss = -pos_score - t.sum(neg_score)
        loss = t.mean(loss)

        return loss

    def prepare_inputs(self, image, y):
        word_image = t.tensor(image, dtype=t.double, device=self.device)
        y_lookup = t.tensor([y], dtype=t.long, device=self.device)
        neg_indexes = np.random.randint(
            0, len(self.neg_dist), size=(len(y), self.neg_samples))  # .flatten()
        neg_indexes = self.neg_dist[neg_indexes]  # .reshape((-1, 5)).tolist()
        neg_lookup = t.tensor(neg_indexes, dtype=t.long, device=self.device)
        return word_image, y_lookup, neg_lookup

    def get_embedding(self, image):
        word_images = t.tensor(image, dtype=t.double, device=self.device)
        input_x = self.layer1(word_images).view(1, -1)
        vI = self.fc1(input_x)
        embeddings = vI.detach().numpy()
        return embeddings

    def save_embedding(self, embed_dict, file_name, device):
        file = open(file_name, encoding='utf8', mode='w')
        file.write("{0} {1}\n".format(len(word2int), self.embed_size))
        for word in embed_dict.keys():
            e = embed_dict[word]
            e = ' '.join([str(x) for x in e])
            file.write("{0} {1}\n".format(word, e))
        file.close()


def generateSG(data, skip_window, batch_size,
               int2word, char2tup, n_chars, n_consonant, n_vowels):
    win_size = skip_window  # np.random.randint(1, skip_window + 1)
    i = win_size
    while True:
        batch_input = []
        batch_output = []
        batch_vec_input = []
        for bi in range(0, batch_size, skip_window * 2):
            context = data[i - win_size: i + win_size + 1]
            target = context.pop(win_size)
            targets = [target] * (win_size * 2)
            batch_input.extend(targets)
            batch_output.extend(context)

            con_mat, vow_mat = word2vec_seperated(char2tup,
                                                  int2word[target], n_chars, n_consonant, n_vowels)
            word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
                (1, 1, n_chars, (n_consonant + n_vowels)))
            batch_vec_input.extend([word_mat] * (win_size * 2))
            i += 1
            if i + win_size + 1 > len(data):
                i = win_size
        batch_vec_input = np.vstack(batch_vec_input)
        yield batch_input, batch_vec_input, batch_output


words = read_file()
words, word2freq = min_count_threshold(words)
# words = subsampling(words, 1e-3)
vocab, word2int, int2word = build_vocab(words)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
print("Words to train: ", len(words))
print("Vocabs to train: ", len(vocab))
print("Unk count: ", word2freq['<unk>'])
int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)
n_chars = 11 + 2
n_epoch = 1
batch_size = 10
skip_window = 1
init_lr = .1
gen = generateSG(list(int_words), skip_window, batch_size,
                 int2word, char2tup, n_chars, n_consonant, n_vowel)

ns_unigrams = np.array(
    ns_sample(word2freq, word2int, int2word, .75), dtype=np.int32)

net = Net(neg_dist=ns_unigrams, embed_size=100,
          vocab_size=len(vocab), device='cpu')
sgd = optimizers.SGD(net.parameters(), lr=init_lr)
start = time.time()
losses = []
grad_time = []
forward_time = []
backward_time = []
step_time = []
start_time = time.time()
steps_per_epoch = (len(int_words) * skip_window) // batch_size
for i in range(steps_per_epoch * n_epoch):
    sgd.zero_grad()
    x1, x2, y = next(gen)
    out = net.forward(x2, x1, y)
    out.backward()
    sgd.step()
    n_words = i * batch_size
    lr = max(.0001, init_lr * (1.0 - n_words /
                               (len(int_words) * skip_window * n_epoch)))
    for param_group in sgd.param_groups:
        param_group['lr'] = lr
    losses.append(out.detach().cpu().numpy())
    if i % (steps_per_epoch // 2) == 0:
        s = "Loss: {0:.4f} lr: {1:.4f} Time Left: {2:.2f}"
        span = (time.time() - start_time)
        print(s.format(np.mean(losses), lr, span))
        start_time = time.time()
        break

del word2int['<unk>']
vocab = list(word2int.keys())
embed_dict = {}
for i in range(len(vocab)):
    word = vocab[i]
    con_mat, vow_mat = word2vec_seperated(
        char2tup, word, n_chars, n_consonant, n_vowel)
    word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
        (1, 1, n_chars, (n_consonant + n_vowel)))
    y = [1]
    em_row = net.get_embedding(word_mat)
    embed_dict[word] = em_row.reshape((-1,))
 

net.save_embedding(embed_dict, "results/w2v_cnn.txt", 'cpu')
