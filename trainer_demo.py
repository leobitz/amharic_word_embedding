from tf_trainer import *
from word2vec import *
from data_handle import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 120
embedding_size = 128
skip_window = 4

words = read_file()
# words = subsampling(words, threshold=1e-3)
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]

vocab_size = len(vocab)
steps_per_batch = len(words) * skip_window // batch_size

int_words = words_to_ints(word2int, words)
print("Final train data: {0}".format(len(words)))
gen = generate_batch_embed(int_words, batch_size, skip_window)

graph = tf.Graph()
with graph.as_default():
    w2v_model = Word2Vec(vocab_size=vocab_size,
                         embed_size=embedding_size,
                         num_sampled=10,
                         batch_size=batch_size,
                         unigrams=unigrams)


trainer = Trainer(train_name="full_200")


trainer.train(graph=graph,
              model=w2v_model,
              gen=gen,
              steps_per_batch=steps_per_batch,
              embed_size=embedding_size,
              epoches=15)
