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
from word2vec_trans import Word2VecTrans
from Word2vec_dense import Word2VecDense
from word2vec_merged import Word2VecMerged
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_new_embedding(oldw2i, neww2i, embeddings):
    new_em = np.ndarray((len(neww2i), embeddings.shape[1]), dtype=np.float32)
    for key in neww2i:
        if key != "<unk>":
            index = oldw2i[key]
            em = embeddings[index]
            new_em[neww2i[key]] = em
    new_em[0] = np.random.rand(50)
    return new_em


batch_size = 120
embedding_size = 75
skip_window = 1
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
n_chars = 11 + 2
n_features = len(char2int)

words = read_file()
unkown_word = "<unk>"
xvocab, xword2int, xint2word = build_vocab(words)

words, word2freq = min_count_threshold(words)
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]
print(word2int['<unk>'])

vocab_size = len(vocab)
steps_per_batch = len(words) * skip_window * 2 // batch_size

int_words = words_to_ints(word2int, words)
print("Final train data: {0}".format(len(words)))
embeddings = np.load("results/seq_encoding.npy")
embeddings = 0.001 * get_new_embedding(xword2int, word2int, embeddings)
# gen = generate_batch_embed(int_words, batch_size, skip_window)
gen = generate_batch_embed_v2(int_words, embeddings, batch_size, skip_window)
# gen = generate_batch_input_dense(int_words, embeddings, batch_size, skip_window, embedding_size)
# gen = generate_batch_rnn_v2(
#     int_words, int2word, char2int, batch_size, skip_window, n_chars, n_features)

graph = tf.Graph()
with graph.as_default():
    # model = Word2VecMerged(vocab_size=vocab_size,
    #                        embed_size=embedding_size,
    #                        num_sampled=5,
    #                        batch_size=batch_size,
    #                        unigrams=unigrams)
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
    model = Word2VecDense(vocab_size=vocab_size,
                          embed_size=embedding_size,
                          num_sampled=5,
                          batch_size=batch_size,
                          unigrams=unigrams)
    model.total_epoches = 10
    model.total_words = len(int_words)
trainer = Trainer(train_name="newdense")


session = trainer.train(graph=graph,
                        model=model,
                        gen=gen,
                        steps_per_batch=steps_per_batch,
                        embed_size=embedding_size,
                        epoches=10)
