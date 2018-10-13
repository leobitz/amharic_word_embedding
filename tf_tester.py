import tensorflow as tf
import numpy as np
import collections
from em_data_gen_v2 import *
import time
import os


class Tester:

    def __init__(self, train_name, max_words=-1):
        self.train_name = train_name
        self._prepare_data(max_words)

    
    def _prepare_data(self, max_words):
        filename = "data/all.txt"
        self.words, self.vocab = get_data(filename, max_words=max_words)
        self.data, self.word2int, self.int2word = build_dataset(self.words)
        self.vocab_size = len(self.vocab)
        print("Words: {0} Vocab: {1}".format(len(self.words), self.vocab_size))
    

    def evaluate(self, graph, w2v_model, gensim_model, model_name):
        with graph.as_default():
            self.saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:

            self.saver.restore(session, model_name)
            embeds = w2v_model.get_embedding()
            gensim_model.set_embeddings(self.word2int, embeds)
            gensim_model.evaluate()
                