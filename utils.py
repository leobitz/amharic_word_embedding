import numpy as np
from numpy.linalg import norm


class Utils:

    def __init__(self, word2int,int2word, embedding):
        self.word2int = word2int
        self.embedding = embedding
        self.int2word = int2word
    
    def cosine_sim(self, word1, word2):
        v1 = self.embedding[self.word2int[word1]]
        v2 = self.embedding[self.word2int[word2]]
        return self.cosine_similarity(v1, v2)
    
    def sorted_sim(self, word1):
        v = self.embedding[self.word2int[word1]]
        sims = self.similarity(self.embedding, v)
        return self.sort_by_similarity(self.word2int, sims)
    
    def solve(self, word1, word2, word3):
        v1 = self.embedding[self.word2int[word1]]
        v2 = self.embedding[self.word2int[word2]]
        v3 = self.embedding[self.word2int[word3]]
        return self.closest_analogy(self.word2int, self.embedding, v1, v2, v3)

    def cosine_similarity(self, u, v):
        dot = v.dot(u)
        norms = norm(u) * norm(v)
        return dot / norms

    def similarity(self, M, v):
        sims = M.dot(v)
        return sims

    def sort_by_similarity(self, word2int, sims):
        sim_with_word = [(word, sims[word2int[word]])
                         for word in word2int.keys()]
        return sorted(sim_with_word, key=lambda t: t[1], reverse=True)[:10]

    def closest_analogy(self, word2int, M, v1, v2, v3):
        v = v1 - v2 + v3
        v = v / norm(v)
        sims = self.similarity(M, v)
        return self.sort_by_similarity(word2int, sims)

    def search_for(self, int2word, M, v):
        sim = self.similarity(M, v)
        argmax = np.argmax(sim)
        word = int2word[argmax]
        return word
    
    def solve_single(self, words):
        v1 = self.embedding[self.word2int[words[0]]]
        v2 = self.embedding[self.word2int[words[1]]]
        v3 = self.embedding[self.word2int[words[2]]]
        v = v1 - v2 + v3
        v = v / norm(v)
        return self.search_for(self.int2word, self.embedding, v)
    
    def evaluate_word_analogy(self, file):
        lines = open(file, encoding='utf8').readlines()
        correct = 0
        total = 0
        for line in lines:
            if line[0] == ':':
                continue
            words = line[:-1].split(' ')
            result = self.solve_single(words)
            if words[-1] == result:
                correct += 1
            total += 1
        return correct / total
