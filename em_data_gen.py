import collections
import random
from tempfile import gettempdir
import zipfile
from data_gen import *
import numpy as np

dg = DataGen()
filename = "data/news.txt"

vocabulary = open(filename, encoding='utf8').read().split()
vocab = list(set(vocabulary))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = len(vocab)
print('Data size', len(vocabulary), ' Vocab Size', vocabulary_size)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.

data_index = 0


def generate_batch(batch_size, num_skips, skip_window, n_chars, n_features):
    #     batch_size = batch_size // 2
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    gru_input = np.ndarray(shape=(batch_size, n_chars, n_features), dtype=np.float32)
    gru_output = np.ndarray(shape=(batch_size, n_chars, n_features), dtype=np.float32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(
        maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
            input_word = reverse_dictionary[context_word]
            g_input = '&' + input_word[:-1]
            input_vec = dg.word2vec2(g_input)
            output_vec = dg.word2vec2(input_word)
            gru_input[i * num_skips + j] = input_vec
            gru_output[i * num_skips + j] = output_vec
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, gru_input, gru_output
