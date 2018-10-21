import tensorflow as tf
import numpy as np
import collections
import time
from word2vec import *
import os


class Tester:

    def __init__(self):
        self.embeddings = None


    def evaluate(self, graph, model, gensim_model, word2int, model_name):
        with graph.as_default():
            self.saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            self.saver.restore(session, model_name)
            embeds = model.get_embedding_v2(session)
            self.embeddings = embeds
            gensim_model.set_embeddings(word2int, embeds)
            gensim_model.evaluate()

    def evaluatev2(self, gensim_model, embeddings):
        self.embeddings = embeddings
        gensim_model.set_embeddings(self.word2int, self.embeddings)
        gensim_model.evaluate()
