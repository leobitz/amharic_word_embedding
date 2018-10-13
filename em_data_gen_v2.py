import collections
import numpy as np
import random


def get_data(name, max_words=-1):
    """Opens a corpus file and returns splited array of words sequence and thier the set of unique words in the corpus

        Example: 

        > get_data("my_corpus.txtx")
        (['hello', 'hi', 'greetings', 'hi'], ['hi', 'hello', 'greetings'])

        Params: 
            name: file path of the corpus
        Returns:
            words - list containing words in the corpus with thier sequence
            vocab - list of the unique words
    """
    text = open(name, encoding='utf8').read()
    if max_words != -1:
        max_chars = 16 * max_words
        text = text[:max_chars]
    words = text.split()
    vocab = list(set(words))

    return words, vocab


def build_dataset(words):
    """Returns int to word index and dictionaries to access words

    Example:

    > build_dataset(['hi', 'hello', 'hi'])
    ([0, 1, 0], {hi: 0, hello: 1}, {0: hi, 1: hello})

    Params:
        words - a list containing words

    Returns:
        data - list of words, words replaced with integers
        word2int - dictionary that maps words to integer index
        int2word - dictionary that maps integers to words
    """

    word2int = {}
    data = []
    for word in words:
        if word not in word2int:
            data.append(len(word2int))
            word2int[word] = len(word2int)
        else:
            data.append(word2int[word])

    int2word = dict(zip(word2int.values(), word2int.keys()))
    return data, word2int, int2word


data_index = 0


def generate_batch(data, batch_size, num_skips, skip_window):
    """Generates batch of input and output pair

    Params:
        data - integer indexed sequence of the corpus words
        batch_size - batch size
        num_skips - how many times to reuse a single word
        skip_window - the number left and right words to consider

    Returns:
        batch - batch of input which is X
        labels - batch of labels for the input Y
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
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
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def build_dataset_v2(sentenses):
    word2int, int2word, data = {}, {}, []
    for sentense in sentenses:
        for word in sentense:
            if word not in word2int:
                int2word[len(int2word)] = word
                word2int[word] = len(word2int)
            data.append(word2int[word])
    return data, word2int, int2word


def ints2words(ints, int2word):
    words = [int2word[i] for i in ints]
    return words


def generate_batch_v2(batch_size, skip_window):
    assert batch_size % skip_window == 0
    sentenses = open('data/all.txt', encoding='utf-8').read().split('*')
    sentenses = [s.strip().split() for s in sentenses]
    data, word2int, int2word = build_dataset_v2(sentenses)
    ci = skip_window  # current_index
    window = 2 * skip_window + 1  # left word right
    while True:
        batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch_index = 0
        while batch_index + skip_window * 2 < batch_size:  # fill the batch inputs
            context = data[ci - skip_window:ci + skip_window + 1]
            target = context[skip_window]
            del context[skip_window]  # remove the target from context words
            context = random.sample(context, skip_window * 2)
            for ic in range(len(context)):
                batch_inputs[batch_index] = context[ic]
                batch_labels[batch_index, 0] = target
                batch_index += 1
            ci += 1
        if len(data) - ci - skip_window < batch_size:
            ci = skip_window
        yield batch_inputs, batch_labels


# gen = generate_batch_v2(10, 2)
# while True:
#     x, y = next(gen)
#     print(x.shape)
