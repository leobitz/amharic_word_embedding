import os
import numpy as np
import tensorflow as tf
import random

tf.set_random_seed(1000)
np.random.seed(1000)
random.seed(1000)

from word2vec_reg import Word2VecReg
from tf_tester import Tester
from tf_trainer import *
from word2vec import *
from word2vec_rnn import *
from gensim_wrapper import *
from data_handle import *
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 500
embedding_size = 128
skip_window = 5

char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()

n_chars = 11 + 2
n_features = len(char2int)

words = read_file()
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]

vocab_size = len(vocab)
steps_per_batch = len(words) // batch_size

int_words = words_to_ints(word2int, words)
print("Final train data: {0}".format(len(words)))

name = "test"
model_name = "log/{0}/model-15".format(name)

gensim_model = GensimWrapper(embedding_size, 0, log=False)

graph = tf.Graph()
with graph.as_default():
    # model = Word2Vec(vocab_size=vocab_size,
    #                    embed_size=embedding_size,
    #                    num_sampled=5,
    #                    batch_size=batch_size,
    #                    unigrams=unigrams)
    # rnn_model = Word2Vec2(vocab_size=vocab_size,
    #                      n_chars=n_chars,
    #                      n_features=n_features,
    #                      embed_size=embedding_size,
    #                      num_sampled=5,
    #                      batch_size=batch_size,
    #                      unigrams=unigrams)
    model = Word2VecReg(vocab_size=vocab_size,
                        embed_size=embedding_size,
                        num_sampled=5,
                        batch_size=batch_size,
                        unigrams=unigrams)
tester = Tester(graph, word2int, model, model_name)
embeddings = tester.embeddings
embeddings = tester.normalize(tester.embeddings)

result = tester.evaluate_v2(gensim_model, embeddings)
utils = Utils(word2int, int2word, embeddings)
# utils.evaluate_word_analogy("data/semantic.txt")
# result = utils.evaluate_word_analogy("data/semantic.txt")
print(result)
print(utils.sorted_sim('አቶ'))
print(utils.sorted_sim('ነው'))
print(utils.sorted_sim('ኢትዮጵያ'))
