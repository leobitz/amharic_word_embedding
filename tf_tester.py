import tensorflow as tf
import numpy as np
import collections
import time
from word2vec import *
import os


class Tester:

    def __init__(self, graph, word2int, model=None, model_name=None):
        self.word2int = word2int
        if model is not None:
            with graph.as_default():
                saver = tf.train.Saver()
            with tf.Session(graph=graph) as session:
                saver.restore(session, model_name)
                embeds = model.get_embedding()
                self.embeddings = embeds

    def evaluate(self, gensim_model):
        gensim_model.set_embeddings(self.word2int, self.embeddings)
        result = gensim_model.evaluate()
        return result

    def normalize(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def evaluate_v2(self, gensim_model, embeddings):
        self.embeddings = embeddings
        gensim_model.set_embeddings(self.word2int, self.embeddings)
        result = gensim_model.evaluate()
        return result
