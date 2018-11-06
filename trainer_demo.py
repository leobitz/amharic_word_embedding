import os
import numpy as np
import tensorflow as tf
import random

tf.set_random_seed(1000)
np.random.seed(1000)
random.seed(1000)

from tf_trainer import *
from word2vec import *
from data_handle import *
from word2vec_rnn import Word2Vec2
from word2vec_reg import Word2VecReg
from word2vec_pre import Word2VecPre
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 512
embedding_size = 128
skip_window = 4
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
n_chars = 11 + 2
n_features = len(char2int)

words = read_file()[:10000]
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]

vocab_size = len(vocab)
steps_per_batch = len(words) * skip_window // batch_size

int_words = words_to_ints(word2int, words)
print("Final train data: {0}".format(len(words)))
embeddings = np.load("results/char_embedding.npy")
# gen = generate_batch_embed(int_words, batch_size, skip_window)
gen = generate_batch_embed_v2(int_words, embeddings, batch_size, skip_window)
# gen = generate_batch_rnn_v2(
#     int_words, int2word, char2int, batch_size, skip_window, n_chars, n_features)

graph = tf.Graph()
with graph.as_default():
    # model = Word2Vec(vocab_size=vocab_size,
    #                  embed_size=embedding_size,
    #                  num_sampled=5,
    #                  batch_size=batch_size,
    #                  unigrams=unigrams)
    # model = Word2Vec2(vocab_size=vocab_size,
    #                   n_chars=n_chars,
    #                   n_features=n_features,
    #                   embed_size=embedding_size,
    #                   num_sampled=5,
    #                   batch_size=batch_size,
    #                   unigrams=unigrams)
    # model = Word2VecReg(vocab_size=vocab_size,
    #                   embed_size=embedding_size,
    #                   num_sampled=10,
    #                   batch_size=batch_size,
    #                   unigrams=unigrams)
    model = Word2VecPre(vocab_size=vocab_size,
                      embed_size=embedding_size,
                      num_sampled=5,
                      batch_size=batch_size,
                      unigrams=unigrams)

trainer = Trainer(train_name="test3")


session = trainer.train(graph=graph,
                        model=model,
                        gen=gen,
                        steps_per_batch=steps_per_batch,
                        embed_size=embedding_size,
                        epoches=20)
# print("starting next")
# gen = generate_batch_embed(int_words, batch_size, skip_window)

# graph = tf.Graph()
# with graph.as_default():
#     model = Word2Vec(vocab_size=vocab_size,
#                      embed_size=embedding_size,
#                      num_sampled=10,
#                      batch_size=batch_size,
#                      unigrams=unigrams)
#     # model = Word2Vec2(vocab_size=vocab_size,
#     #                   n_chars=n_chars,
#     #                   n_features=n_features,
#     #                   embed_size=embedding_size,
#     #                   num_sampled=5,
#     #                   batch_size=batch_size,
#     #                   unigrams=unigrams)
#     # model = Word2VecReg(vocab_size=vocab_size,
#     #                   embed_size=embedding_size,
#     #                   num_sampled=5,
#     #                   batch_size=batch_size,
#     #                   unigrams=unigrams)

# trainer = Trainer(train_name="test")


# session = trainer.train(graph=graph,
#                         model=model,
#                         gen=gen,
#                         steps_per_batch=steps_per_batch,
#                         embed_size=embedding_size,
#                         epoches=40)