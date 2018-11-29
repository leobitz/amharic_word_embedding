from tf_trainer_rnn import *
from gensim_wrapper import *
import os
from data_handle import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def remove_large_words(words, max_len):
    new_list = []
    for word in words:
        if len(word) <= max_len:
            new_list.append(word)
    return new_list
def get_new_embedding(oldw2i, neww2i, embeddings):
    new_em = np.ndarray((len(neww2i), embeddings.shape[1]), dtype=np.float32)
    for key in neww2i:
        index = oldw2i[key]
        em = embeddings[index]
        new_em[neww2i[neww2i[key]]] = em
    return new_em


batch_size = 500
embedding_size = 128
skip_window = 5
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()

n_chars = 11 + 2
n_features = len(char2int)

words = read_file()  # [:1000]
words = remove_large_words(words, n_chars)
vocab, word2int, int2word = build_vocab(words)

word2freq = get_frequency(words, word2int, int2word)
unigrams = [word2freq[int2word[i]] for i in range(len(word2int))]

vocab_size = len(vocab)
steps_per_batch = len(words) // batch_size

int_words = words_to_ints(word2int, words)
print("Final train data: {0}".format(len(words)))
gen = generate_batch_rnn_v2(
    int_words, int2word, char2int, batch_size, skip_window, n_chars, n_features)

graph = tf.Graph()
with graph.as_default():
    model = Word2Vec2(vocab_size=vocab_size,
                      n_chars=n_chars,
                      n_features=n_features,
                      embed_size=embedding_size,
                      num_sampled=5,
                      batch_size=batch_size,
                      unigrams=unigrams)
trainer = RnnTrainer(train_name="full_200",
                     batch_size=batch_size,
                     skip_window=skip_window)

trainer.train(graph, model, gen, steps_per_batch, embedding_size, epoches=10)
