import tensorflow as tf
import numpy as np
import collections
from em_data_gen_v2 import *
import time
from word2vec import *
import os


class Tester:

    def __init__(self, max_words=-1):
        self._prepare_data(max_words)
        self.embeddings = None

    def _prepare_data(self, max_words):
        filename = "data/news.txt"
        self.words, self.vocab = get_data(filename, max_words=max_words)
        self.data, self.freq, self.word2int, self.int2word = build_dataset(self.words)
        self.vocab_size = len(self.vocab)
        print("Words: {0} Vocab: {1}".format(len(self.words), self.vocab_size))

    def evaluate(self, gensim_model, model_name, batch_size, embed_size):
        graph = tf.Graph()
        with graph.as_default():
            wv_model = Word2Vec(vocab_size=self.vocab_size,
                                embed_size=embed_size,
                                batch_size=batch_size,
                                wordfreq=None
                                )
            self.saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            self.saver.restore(session, model_name)
            embeds = wv_model.get_embedding_v2(session)
            self.embeddings = embeds
            gensim_model.set_embeddings(self.word2int, embeds)
            gensim_model.evaluate()

    def evaluatev2(self, gensim_model, embeddings):
        self.embeddings = embeddings
        gensim_model.set_embeddings(self.word2int, self.embeddings)
        gensim_model.evaluate()
