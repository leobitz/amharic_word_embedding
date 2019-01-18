import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
import time
from data_handle import *
from torch.autograd import Variable
from w2vtorch_high import Net

def save_result(step):
    vocab = list(word2int.keys())
    result_dicts = []
    for i in range(len(vocab)):
        word = vocab[i]
        if '<unk>' in word:
            continue
        con_mat, vow_mat = word2vec_seperated(
            char2tup, word, n_chars, n_consonant, n_vowel)
        word_mat = np.concatenate([con_mat, vow_mat], axis=1).reshape(
            (1, 1, n_chars, (n_consonant + n_vowel)))
        x_index = word2int[word]
        result = net.get_embedding(word_mat, [x_index])
        for j in range(len(result)):
            if len(result_dicts) < j + 1:
                result_dicts.append({})
            result_dicts[j][word] = result[j].reshape((-1,))

    for counter, rdict in enumerate(result_dicts):
        net.save_embedding(rdict, "results/w2v_high_{0}_{1}.txt".format(counter, step), device)


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
# print("Unk count: ", word2freq['<unk>'])
int_words = words_to_ints(word2int, words)
int_words = np.array(int_words, dtype=np.int32)
n_chars = 11 + 2
n_epoch = 5
batch_size = 10
skip_window = 3
init_lr = .1
gen = generateSG(list(int_words), skip_window, batch_size,
                 int2word, char2tup, n_chars, n_consonant, n_vowel)

ns_unigrams = np.array(
    ns_sample(word2freq, word2int, int2word, .75), dtype=np.int32)
device = 'cpu'
net = Net(neg_dist=ns_unigrams, embed_size=100,
          vocab_size=len(vocab), device=device)
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
    out  = net.forward(x2, x1, y)
    out.backward() 
    sgd.step()
    n_words = i * batch_size
    lr = max(.0001, init_lr * (1.0 - n_words /
                               (len(int_words) * skip_window * n_epoch)))
    for param_group in sgd.param_groups:
        param_group['lr'] = lr
    losses.append(out.detach().cpu().numpy())
    if i % (steps_per_epoch // 100) == 0:
        # print(seq_prob, vI_prop)
        s = "Loss {0:.4f} lr: {1:.4f} Time Left: {2:.2f}"
        span = (time.time() - start_time)
        print(s.format(np.mean(losses), lr, span))
        start_time = time.time()
    if i > 0 and i % steps_per_epoch == 0 :
        save_result(i//steps_per_epoch)

save_result(n_epoch)
