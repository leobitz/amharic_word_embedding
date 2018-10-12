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

    words = open(name, encoding='utf8').read().split()
    if max_words != -1:
        words = words[:max_words] # upto max_words only
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
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
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
