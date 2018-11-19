import numpy as np
from numpy.linalg import norm
from data_handle import *


class Utils:

    def __init__(self, word2int, int2word, embedding):
        self.word2int = word2int
        self.embedding = embedding
        self.int2word = int2word

    def cosine_similarity(self, word1, word2):
        v1 = self.embedding[self.word2int[word1]]
        v2 = self.embedding[self.word2int[word2]]
        return self.cosine_sim(v1, v2)

    def sorted_sim(self, word1):
        v = self.embedding[self.word2int[word1]]
        sims = self.similarity(self.embedding, v)
        return self.sort_by_similarity(self.word2int, sims)

    def solve(self, word1, word2, word3):
        v1 = self.embedding[self.word2int[word1]]
        v2 = self.embedding[self.word2int[word2]]
        v3 = self.embedding[self.word2int[word3]]
        return self.levy_solve(v1, v2, v3)
        # return self.closest_analogy(self.word2int, self.embedding, v1, v2, v3)

    def cos_sim_whole(self, v):
        return np.dot(self.embedding, v)

    def levy_solve(self, v1, v2, v3):
        r1 = self.cos_sim_whole(v1)
        r2 = self.cos_sim_whole(v2)
        r3 = self.cos_sim_whole(v3)
        argmax = np.argmax(r1 - r2 + r3)
        return int2word[argmax]

    def cosine_sim(self, u, v):
        dot = v.dot(u)
        norms = norm(u) * norm(v)
        return dot / norms

    def similarity(self, M, v):
        sims = M.dot(v)
        return sims

    def sort_by_similarity(self, word2int, sims, top=10):
        sim_with_word = [(word, sims[word2int[word]])
                         for word in word2int.keys()]
        return sorted(sim_with_word, key=lambda t: t[1], reverse=True)[:top]

    def closest_analogy(self, word2int, M, v1, v2, v3):
        v = self.normalize(v1 - v2 + v3)
        sims = self.similarity(M, v)
        return self.sort_by_similarity(word2int, sims)

    def search_for(self, M, v):
        sim = self.similarity(M, v)
        argmax = np.argmax(sim)
        word = self.int2word[argmax]
        return word

    def solve_single(self, words):
        v1 = self.embedding[self.word2int[words[0]]]
        v2 = self.embedding[self.word2int[words[1]]]
        v3 = self.embedding[self.word2int[words[2]]]
        v = v2 - v1 + v3
        return self.search_for(self.embedding, v)

    def evaluate_word_analogy(self, file):
        W = self.embedding
        lines = open(file, encoding='utf8').readlines()
        correct = 0
        total = 0
        int_pairs = []
        try:
            for line in lines:
                if line[0] == ':':
                    continue
                words = line[:-1].split(' ')
                int_pairs.append([word2int[word] for word in words])
        except:
            pass
        ind = np.array(int_pairs, dtype=np.int32)

        V1, V2, V3 = W[ind[:, 0]], W[ind[:, 1]], W[ind[:, 2]]
        V = V2 - V1 + V3
        R = np.dot(W, V.T)
        maxR = np.argmax(R, 0)
        corrects = np.sum(maxR == ind[:, 3])
        return corrects * 100 / len(int_pairs)

    def normalize(self, v):
        return v / norm(v)


# words = read_file()
# vocab, word2int, int2word = build_vocab(words)

# words = read_file(filename='data/all.txt')
# vocab, word2int, int2word = build_vocab(words)
# word2freq = get_frequency(words, word2int, int2word)

# words = min_count_threshold(words, word2freq, 5)
# vocab, word2int, int2word = build_vocab(words)
# print(len(vocab))

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
#     embeddings = np.ndarray((vocab_size+1, embed_size), dtype=np.float32)
#     for i in range(vocab_size):
#         line = lines[i+1].split(delimiter)[:-1]
#         word = line[0]
#         if word in word2int:
#             wordvec = np.array([float(j) for j in line[1:]])
#             embeddings[word2int[word]] = wordvec
#     return embeddings

# fast = parseVec('results/model.vec', ' ')
# fast = normalize(fast)

# util = Utils(word2int, int2word, fast)
# print(util.evaluate_word_analogy('data/analogies.txt'))
# from gensim_wrapper import *
# gw = GensimWrapper('data/all.txt', 100, 0)
# # embeddings = normalize(np.load('results/char_embedding.npy'))
# gw.set_embeddings(word2int, fast)
# print(gw.evaluate())
# util = Utils(word2int, int2word, embeddings)
# print(util.evaluate_word_analogy('data/analogies.txt'))
