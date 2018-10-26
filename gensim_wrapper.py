import gensim
import logging
import collections


class GensimWrapper:

    def __init__(self, embed_size=128, iter=5, log=False):
        if log:
            logging.basicConfig(
                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.embed_size = embed_size
        self.iter = iter
        self._prepare_data()
        self._create_model()

    def _prepare_data(self):
        sentenses = open('data/news.txt', encoding='utf-8').read().split('*')
        self.data = [s.strip().split() for s in sentenses]

    def _create_model(self):
        self.model = gensim.models.Word2Vec(
            self.data, size=self.embed_size, iter=self.iter, min_count=1)

    def evaluate_anomaly(self):
        lines = open('data/anomaly.txt', encoding='utf-8').readlines()
        correct = 0
        for line in lines:
            vals = line[:-1].split(' ')
            index = int(vals[-1])
            pred = self.model.doesnt_match(vals[:4])
            if pred == vals[index]:
                correct += 1
        return correct * 100 / len(lines)

    def evaluate_semantic_analogy(self):
        result = self.model.accuracy('data/semantic.txt')
        return result

    def evaluate_synatctic_analogy(self):
        result = self.model.accuracy('data/syntax.txt')
        return result

    def evaluate(self):
        anomaly = self.evaluate_anomaly()
        semantic = self.evaluate_semantic_analogy()[0]
        syntactic = self.evaluate_synatctic_analogy()[0]
        semantic = len(semantic['correct']) * 100 / (len(semantic['correct']) + len(semantic['incorrect']))
        syntactic = len(syntactic['correct']) * 100 / (len(syntactic['correct']) + len(syntactic['incorrect']))
        evaluation = {"anomaly": anomaly,
                      "semantic": semantic, "syntactic": syntactic}
        return evaluation

    def set_embeddings(self, word2int, embeddings):
        """Transfers the word embedding learned bu tensorflow to gensim model
        Params:
            gensim_model - un untrained gensim_model
            word2int - dictionary that maps words to int index
            embedding - a new learned embeddings by tensorflow
        """
        self.model.wv.init_sims()
        for gindex in range(len(self.model.wv.index2word)):
            gword = self.model.wv.index2word[gindex]
            index = word2int[gword]
            embedding = embeddings[index]
            # print(gindex, gword, type(self.model.wv.vectors_norm))
            self.model.wv.vectors_norm[gindex] = embedding
            # self.model.wv.vectors[gindex] = embedding
