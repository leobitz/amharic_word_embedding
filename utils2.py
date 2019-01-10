import numpy as np
from numpy.linalg import norm
from data_handle import *


class Utils:

    def __init__(self, word2int, int2word, emb1, emb2):
        self.word2int = word2int
        self.int2word = int2word
        self.emb1 = emb1
        self.emb2 = emb2

    def dotM(self, M, v):
        return np.dot(M, v)

    def get_sorted(self, cos_proximity):
        indexes = np.argsort(-cos_proximity)
        sim_words = [self.int2word[i] for i in indexes[:3]]
        return sim_words

    def get_vectors(self, M, words):
        v1 = M[self.word2int[words[0]]]
        v2 = M[self.word2int[words[1]]]
        v3 = M[self.word2int[words[2]]]
        return [v1, v2, v3]

    def calc_analogy_vec(self, M, vecs):
        v = vecs[1] - vecs[0] + vecs[2]
        cos_prox = self.dotM(M, v)
        sim_words = self.get_sorted(cos_prox)
        return sim_words

    def calc_analogy_vec_v2(self, M1, M2, V1, V2):
        v1 = V1[1] - V1[0] + V1[2]
        v2 = V2[1] - V2[0] + V2[2]
        cos_prox = self.dotM(M1, v1) + self.dotM(M2, v2)
        sim_words = self.get_sorted(cos_prox)
        return sim_words

    def solve(self, words):
        oov = False
        for word in words:
            if word not in self.word2int:
                oov = True
        if oov:
            return None
        V1 = self.get_vectors(self.emb1, words)
        V2 = self.get_vectors(self.emb2, words)
        sim_words = self.calc_analogy_vec_v2(self.emb1, self.emb2, V1, V2)
        return sim_words


# words = read_file()
# vocab, word2int, int2word = build_vocab(words)

# words = read_file(filename='data/all.txt')
# vocab, word2int, int2word = build_vocab(words)
# word2freq = get_frequency(words, word2int, int2word)

# words, word2freq = min_count_threshold(words, 5)
# vocab, word2int, int2word = build_vocab(words)


# int_words = words_to_ints(word2int, words)
# word2freq = get_frequency(words, word2int, int2word)
# char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
# ns_unigrams = ns_sample(word2freq, word2int, int2word, .75)
# n_chars = 11 + 2
# n_features = len(char2int)
# batch_size = 120
# embed_size = 100
# skip_window = 5
# # int_words = words_to_ints(word2int, words)
# # word2freq = get_frequency(words, word2int, int2word)
# # char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
# # ns_unigrams = ns_sample(word2freq, word2int, int2word, .75)
# # n_chars = 11 + 2
# # n_features = len(char2int)
# # batch_size = 120
# # embed_size = 128
# # skip_window = 5
# def parseVec(file, delimiter):
#     lines = open(file, encoding='utf8').readlines()
#     vocab_size, embed_size = [int(s) for s in lines[0].split()]
#     vocab_size  = min(len(word2int), vocab_size)
#     embeddings = np.ndarray((vocab_size, embed_size), dtype=np.float64)
#     for i in range(1, vocab_size):
#         try:
#             line = lines[i][:-1].split(delimiter)
#             word = line[0]
#             if word in word2int:
#                 wordvec = np.array([np.float64(j) for j in line[1:] if j != ''])
#                 embeddings[word2int[word]] = wordvec
#         except Exception as e:
#             print(lines[i])
#             print(e)
#     return embeddings

# sem = normalize(parseVec('results/w2v.txt', ' ', True))
# syn = normalize(parseVec('results/w2v_torch.txt_', ' ', False))

# result = evaluate(word2int, sem)
# print(result)
# result = evaluate(word2int, syn)
# print(result)
# util = Utils(word2int, int2word, sem, syn)
# lines = open('data/newan2.txt', encoding='utf-8').readlines()
# acc = np.array([0, 0])
# total = np.array([0, 0])
# ind = 0
# for line in lines[1:]:
#     if ':' in line[:-1]:
#         ind = 1
#         continue
#     ws = line[:-1].split()
#     result = util.solve(ws)
#     if result is None:
#         continue
#     if ws[-1] in result:
#         acc[ind] += 1
#     total[ind] += 1
# print(acc * 100 / total)
