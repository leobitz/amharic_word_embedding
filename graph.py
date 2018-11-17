import numpy as np
from data_handle import *
import matplotlib.pyplot as plt
np.random.seed(10)


class Node:

    def __init__(self, word, index, vec):
        self.word = word
        self.index = index
        self.vec = vec
        self.lefts = {}
        self.rights = []

    def addLeft(self, node):
        if node.word not in self.lefts:
            self.lefts[node.word] = [node, 0]
        self.lefts[node.word] = [self.lefts[node.word]
                                 [0], (self.lefts[node.word][1] + 1)]

    def addRight(self, node):
        self.rights.append(node)

    def calcRank(self):
        vec = np.zeros_like(self.vec, dtype=np.float32)
        i = 0
        for key in self.lefts.keys():
            node = self.lefts[key][0]
            vec += (node.vec * self.lefts[key][1] / len(node.lefts))
            i += self.lefts[key][1]
        self.vec  = vec ** .75



text = "I love deep learning in NLP . deep learning is method to work on NLP . NLP is cool but mostly I love deep ."
words = text.split(' ')
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)
int_words = words_to_ints(word2int, words)


mapping = {}
for word in words:
    if word not in mapping:
        vec = np.random.uniform(-1, 1, size=(3, ))
        mapping[word] = Node(word, word2int[word], vec)

for i in range(len(words) - 1):
    prev_word = words[i]
    word = words[i + 1]
    mapping[word].addLeft(mapping[prev_word])
    mapping[prev_word].addLeft(mapping[word])

nodes = list(mapping.values())
print(vec)
for i in range(100):
    for node in nodes:
        node.calcRank()
    print(node.vec)
xx, yy = [], []
for key in mapping:
    x, y, z = mapping[key].vec
    # print(x, y)
    # print(key)
    xx.append(x)
    yy.append(z)
xx = range(len(xx))
plt.scatter(xx, yy)
for key in mapping:
    x, y, z = mapping[key].vec
    x = xx[word2int[key]]
    plt.annotate(key, (x, z))
plt.show()
