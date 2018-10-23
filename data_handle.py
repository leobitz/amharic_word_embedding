import collections
from collections import Counter
import numpy as np
import random


def read_file(filename='data/news.txt'):
    return open(filename, encoding='utf8').read().strip().split(' ')


def words_to_ints(word2int, words):
    return [word2int[word] for word in words]


def get_frequency(words, word2int, int2word):
    word2freq = {}
    for word in words:
        if word not in word2freq:
            word2freq[word] = 0
        word2freq[word] += 1.0
    return word2freq


def build_vocab(words):
    word2int = {}
    int2word = {}
    for word in words:
        if word not in word2int:
            index = len(word2int)
            int2word[index] = word
            word2int[word] = index
    vocab = list(word2int.keys())
    return vocab, word2int, int2word


def build_charset():
    charset = open('data/charset.txt', encoding='utf-8').readlines()
    n_consonant = len(charset)
    n_vowel = 10
    char2int, int2char, char2tup, tup2char = {}, {}, {}, {}
    j = 0
    for k in range(len(charset)):
        row = charset[k][:-1].split(',')
        for i in range(len(row)):
            char2tup[row[i]] = (k, i)
            int2char[j] = row[i]
            char2int[row[i]] = j
            tup = "{0}-{1}".format(k, i)
            tup2char[tup] = row[i]
            j += 1
    return char2int, int2char, char2tup, tup2char, n_consonant, n_vowel


def word2vec_seperated(char2tup, word, max_word_len, n_consonant, n_vowel):
    cons = np.zeros((max_word_len, n_consonant), dtype=np.float32)
    vowel = np.zeros((max_word_len, n_vowel), dtype=np.float32)
    for i in range(len(word)):
        char = word[i]
        t = char2tup[char]
        cons[i][t[0]] = 1
        vowel[i][t[1]] = 1
    con, vow = char2tup[' ']
    cons[i + 1:, con] = 1
    vowel[i + 1:, vow] = 1
    vec = np.concatenate([cons, vowel], axis=1)
    return vec


def word2vec(char2int, word, max_word_len):
    max_n_char = len(char2int)
    vec = np.zeros((max_word_len, max_n_char), dtype=np.float32)
    for i in range(len(word)):
        char = word[i]
        t = char2int[char]
        vec[i][t] = 1
    spacei = char2int[' ']
    vec[i + 1:, spacei] = 1
    return vec


def one_hot(n, size):
    v = np.zeros((size,))
    v[n] = 1
    return v


def one_hot_decode(int2word, vec):
    indexes = np.argmax(vec, axis=1)
    words = []
    for i in indexes:
        words.append(int2word[i])
    return words


def sentense_to_vec(words):
    vecs = []
    for w in words:
        vecs.append(word2vec(w))
    vec = np.concatenate(vecs)
    return vec


def generate_batch_embed(data, batch_size, skip_window):
    assert batch_size % skip_window == 0
    ci = skip_window  # current_index
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window * 2):  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            # remove the target from context words
            target = context.pop(skip_window)
            # context = random.sample(context, skip_window * 2)
            batch_inputs[batch_index:batch_index +
                         skip_window * 2] = context
            batch_labels[batch_index:batch_index + skip_window * 2, 0] = target
            ci += 1
        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        yield batch_inputs, batch_labels


def generate_word_images(words, char2int, batch_size):
    targets, target_inputs = [], []
    for word in words:
        target = word + '&'
        target_input = '&' + target
        targets.append(target)
        target_inputs.append(target_input)
    batch = 0
    n_batchs = len(words) // batch_size
    while True:
        batch_targets = targets[batch:batch + batch_size]
        batch_target_ins = target_inputs[batch:batch + batch_size]
        batch_words = words[batch:batch + batch_size]
        batch_inputs = []
        batch_outputs = []
        batch_raw_inputs = []
        for i in range(batch_size):
            target = word2vec(char2int, batch_targets[i], 13)
            target_in = word2vec(char2int, batch_target_ins[i], 13)
            word = word2vec(char2int, batch_words[i],  13)
            batch_inputs.append(target)
            batch_outputs.append(target_in)
            batch_raw_inputs.append(word)
        batch_inputs = np.stack(batch_inputs)#.reshape((batch_size, 13, 309, 1))
        batch_outputs = np.stack(batch_outputs)#.reshape((batch_size, 13, 309, 1))
        batch_raw_inputs = np.stack(batch_raw_inputs).reshape(
            (batch_size, 13, 309, 1))
        yield [batch_raw_inputs, batch_inputs], batch_outputs
        batch += 1
        if batch == n_batchs:
            batch = 0


def generate_batch_rnn_v2(data, int2word, char2int, batch_size, skip_window, n_chars, n_features):
    assert batch_size % skip_window == 0
    ci = skip_window  # current_index
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        rnn_inputs = np.ndarray(
            shape=(batch_size, n_chars, n_features), dtype=np.float32)
        rnn_outputs = np.ndarray(
            shape=(batch_size, n_chars, n_features), dtype=np.float32)
        batch_index = 0
        for batch_index in range(0, batch_size, skip_window * 2):  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            # remove the target from context words
            target = context.pop(skip_window)
            # context = random.sample(context, skip_window * 2)
            batch_inputs[batch_index:batch_index +
                         skip_window * 2] = context
            batch_labels[batch_index:batch_index + skip_window * 2] = target
            ci += 1

        for rnn_i in range(batch_size):
            a, b = batch_inputs[rnn_i], batch_labels[rnn_i]
            context_word = '&' + int2word[a] + '&'
            target_word = int2word[b] + '&'
            context_vec = word2vec(char2int, context_word, n_chars)
            target_vec = word2vec(char2int, target_word, n_chars)
            rnn_inputs[rnn_i] = context_vec
            rnn_outputs[rnn_i] = target_vec

        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        batch_labels = batch_labels.reshape((-1, 1))
        yield batch_inputs, batch_labels, rnn_inputs, rnn_outputs


def subsampling(int_words, threshold=1e-5):
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {
        word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    train_words = [word for word in int_words if random.random()
                   < (1 - p_drop[word])]
    return train_words
