import os
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(1000)
np.random.seed(1000)
random.seed(1000)

from word2vec_reg import Word2VecReg
from tf_tester import Tester
from tf_trainer import *
from word2vec import *
from word2vec_rnn import *
from word2vec_pre import Word2VecPre
from word2vec_trans import Word2VecTrans
from Word2vec_dense import Word2VecDense
from word2vec_merged import Word2VecMerged
from gensim_wrapper import *
from data_handle import *
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()

n_chars = 11 + 2
n_features = len(char2int)

words = read_file()
words, word2freq = min_count_threshold(words)
vocab, word2int, int2word = build_vocab(words)
print(len(vocab))
# word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]

batch_size = len(vocab)
embedding_size = 75
skip_window = 5

vocab_size = len(vocab)
steps_per_batch = len(words) // batch_size
metrics = {}
int_words = words_to_ints(word2int, words)

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
    #                     embed_size=embedding_size,
    #                     num_sampled=5,
    #                     batch_size=batch_size,
    #                     unigrams=unigrams)
    model = Word2VecDense(vocab_size=vocab_size,
                        embed_size=embedding_size,
                        num_sampled=5,
                        batch_size=batch_size,
                        unigrams=unigrams)
with tf.Session(graph=graph) as session:
    tester = Tester(graph, session, word2int, model)
    gensim_model = GensimWrapper('data/news.txt', embedding_size, 0, log=False)
    name = "newdense"
    for ti in range(0, 6):
        model_name = "log/{0}/model-{1}".format(name, ti)
        seq_emb = np.load('results/seq_encoding.npy')
        inputs = np.array(list(word2int.values()), dtype=np.int32)
        embed = model.get_embed_2(session, seq_emb[inputs], inputs)
        embeddings = tester.restore(model_name)
        # cons, tars = tester.restore(model_name, seq_emb)
        # embeddings = np.concatenate([embeddings, .0357 * seq_emb], axis=1)
        # seq_norm = np.mean(np.linalg.norm(embeddings))
        # print(seq_norm)
        # seq_norm = np.mean(np.linalg.norm(seq_emb))
        # # print(seq_norm)
        # em_norm = np.mean(np.linalg.norm(embeddings))
        # embeddings =  seq_emb / seq_norm + embeddings/em_norm
        # embeddings = embeddings + 0.033 * seq_emb
        # cons, tars = model.get_embedding(session, seq_emb)
        embeddings_normal = normalize(embeddings)
        # for kki in range(0, batch_size)
        result = tester.evaluate_v2(gensim_model, embed)
        result['average'] = (result['semantic'] + result['syntactic']) / 2
        print(result)
        for key in result:
            if key not in metrics:
                metrics[key] = []
            metrics[key] += [result[key]]

for key in metrics:
    plt.plot(range(len(metrics[key])), metrics[key], label=key)
plt.xlabel("Epoch")
plt.ylabel("Accouracy")
plt.legend()
plt.show()
# utils = Utils(word2int, int2word, embeddings)
# utils.evaluate_word_analogy("data/semantic.txt")
# result = utils.evaluate_word_analogy("data/semantic.txt")
# print(result)
# print(utils.sorted_sim('አቶ'))
# print(utils.sorted_sim('ነው'))
# print(utils.sorted_sim('ኢትዮጵያ'))
