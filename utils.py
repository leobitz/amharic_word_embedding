import numpy as np
from numpy.linalg import norm


class Utils:

    def __init__(self, word2int, embedding):
        self.word2int = word2int
        self.embedding = embedding
    
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
