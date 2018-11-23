import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from keras.layers import Concatenate, Flatten
from keras.layers import GRU, Conv2D, MaxPooling2D, Embedding
from keras.layers import Input, Reshape, Dot, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
# from keras.utils.vis_utils import plot_model
import keras
import keras.backend as K
from data_handle import *
from gensim_wrapper import *
from utils import *
import gensim
import random
import numpy as np
import tensorflow as tf
import gc
from training_handler import *
seed_val = 1000
random.seed(seed_val)
np.random.seed(seed_val)
tf.set_random_seed(seed_val)
batch_size = 100
train_batches = 20000


def conv_model_multi(vocab_size, n_seq, embed_size, n_consonant, n_vowels, n_units):
    root_word_input = Input(
        shape=(seq_len,), dtype='int32', name="root_word_input")

    x = Embedding(vocab_size, embed_size, input_length=n_seq)(root_word_input)
    _, state_h = GRU(n_units, return_sequences=True,
                     return_state=True, activation='relu')(x)

    consonant_decoder_inputs = Input(
        shape=(None, n_consonant), name="target_consonant")
    consonant_decoder_gru = GRU(
        n_units, return_sequences=True, return_state=True,  name="consonant_decoder_gru")
    consonant_decoder_outputs, _ = consonant_decoder_gru(
        consonant_decoder_inputs, initial_state=state_h)

    vowel_decoder_inputs = Input(shape=(None, n_vowels), name="vowel_input")
    vowel_decoder_gru = GRU(n_units, return_sequences=True,
                            return_state=True, name="vowl_decoder_gru")
    vowel_decoder_outputs, _ = vowel_decoder_gru(
        vowel_decoder_inputs, initial_state=state_h)

    consonant_decoder_dense = Dense(
        n_consonant, activation='softmax', name="consonant_output")
    consonant_decoder_outputs = consonant_decoder_dense(
        consonant_decoder_outputs)

    vowel_decoder_dense = Dense(
        n_vowels, activation='softmax', name="vowel_output")
    vowel_decoder_outputs = vowel_decoder_dense(vowel_decoder_outputs)

    main_model = Model([root_word_input, consonant_decoder_inputs, vowel_decoder_inputs], [
                       consonant_decoder_outputs, vowel_decoder_outputs])

    encoder_model = Model(root_word_input, state_h)

    decoder_state_input_h = Input(shape=(n_units,))

    consonant_decoder_outputs, state_h = consonant_decoder_gru(
        consonant_decoder_inputs, initial_state=decoder_state_input_h)
    consonant_decoder_outputs = consonant_decoder_dense(
        consonant_decoder_outputs)

    vowel_decoder_outputs, state_h = vowel_decoder_gru(
        vowel_decoder_inputs, initial_state=decoder_state_input_h)
    vowel_decoder_outputs = vowel_decoder_dense(vowel_decoder_outputs)

    decoder_model = Model([consonant_decoder_inputs, vowel_decoder_inputs, decoder_state_input_h], [
                          consonant_decoder_outputs, vowel_decoder_outputs, state_h])

    return main_model, encoder_model, decoder_model


words = read_file(filename='data/news.txt')
vocab, word2int, int2word = build_vocab(words)
word2freq = get_frequency(words, word2int, int2word)

int_words = words_to_ints(word2int, words)
word2freq = get_frequency(words, word2int, int2word)
char2int, int2char, char2tup, tup2char, n_consonant, n_vowel = build_charset()
ns_unigrams = ns_sample(word2freq, word2int, int2word, .75)
n_chars = 11 + 2
n_features = len(char2int)
batch_size = 100
embed_size = 50
skip_window = 5
seq_len = 5
rnn_n_units = 64
epoches = 2
vocab_size = len(vocab)
n_batches = len(words) // batch_size
save_on_every = 100

main_model, encoder_model, decoder_model = conv_model_multi(
    vocab_size, seq_len, embed_size, n_consonant, n_vowel, rnn_n_units)
adam = keras.optimizers.Nadam(0.001)
main_model.compile(
    optimizer=adam, loss="categorical_crossentropy", metrics=['acc'])
# main_model.summary()
gen = generate_for_char_langauge(words, int_words, int2word, char2tup, batch_size=batch_size,
                                 n_chars=13, n_consonant=n_consonant, n_vowels=10, seq_length=seq_len)


model_name = "lm_char"
tag_name = "2_256"

trainer = TrainingHandler(main_model, model_name)
trainer.train(tag_name, gen, epoches, 
              n_batches, save_on_every,
              save_model=True)
