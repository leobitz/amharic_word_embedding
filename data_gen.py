import numpy as np

class DataGen:
    
    def __init__(self, percent=100):
        self.n_consonant = 0
        self.n_vowel = 10
        self.corpus = open('data/news.txt', encoding='utf8').read()
        self.words = self.corpus.split(' ')
        width = int(len(self.words) * percent/100)
        self.words = self.words[:width]
        self.char2int, self.int2char, self.char2tup, self.tup2char = self.load_charset()
        self.vocab, self.word2int, self.int2word = self.get_vocab(self.words)
        self.max_word_len = 11
        del self.corpus
    
    def get_vocab(self, words):
        word2int = {}
        int2word = {}
        vocab = list(sorted(set(words)))
        for word in vocab:
            int2word[len(word2int)] = word
            word2int[word] = len(word2int)
        return vocab, word2int, int2word
    
    def load_charset(self):
        charset = open('data/charset.txt', encoding='utf-8').readlines()
        self.n_consonant = len(charset)
        char2int, int2char, char2tup, tup2char = {}, {}, {}, {}
        j = 0
        for k in range(len(charset)):
            row = charset[k][:-1].split(',')
            for i in range(len(row)):
                char2tup[row[i]] = (k, i)
                int2char[j] = row[i]
                char2int[row[i]] = j
                tup = "{0}-{1}".format(k, i)
                tup2char[tup] = row[i]
                j += 1
        return char2int, int2char, char2tup, tup2char
        
    
    def word2vec(self, word):
        cons = np.zeros((self.max_word_len, self.n_consonant), dtype=np.float32)
        vowel = np.zeros((self.max_word_len, self.n_vowel), dtype=np.float32)
        for i in range(len(word)):
            char = word[i]
            t = self.char2tup[char]
            cons[i][t[0]] = 1
            vowel[i][t[1]] = 1
        con, vow = self.char2tup[' ']
        cons[i+1:, con] = 1
        vowel[i+1:, vow] = 1
        vec = np.concatenate([cons, vowel], axis=1)
        return vec

    def word2vec2(self, word):
        max_n_char = len(self.char2int)
        vec = np.zeros((self.max_word_len, max_n_char), dtype=np.float32)
        for i in range(len(word)):
            char = word[i]
            t = self.char2int[char]
            vec[i][t] = 1
        spacei = self.char2int[' ']
        vec[i+1:, spacei] = 1
        return vec
    
    def one_hot(self, n, size):
        v = np.zeros((size,))
        v[n] = 1
        return v
    
    def one_hot_decode(self, vec):
        indexes = np.argmax(vec, axis=1)
        words = []
        for i in indexes:
            words.append(self.int2word[i])
        return words
            
        
    
    def sentense_to_vec(self, words):
        vecs = []
        for w in words:
            vecs.append(self.word2vec(w))
        vec = np.concatenate(vecs)
        return vec
    

    def gen(self, batch_size=100, n_batches=-1, windows_size=4):
        batch = 0
        n_words = len(self.words)
        if n_batches > 0:
            n_words = batch_size * n_batches
        c_word = windows_size // 2
        while True:
            x = []
            y = []
            for i in range(batch_size):
                j = c_word - windows_size // 2
                k = c_word + windows_size // 2 + 1
                context = self.words[j:k]
                target = context.pop(windows_size//2)
                vec = self.sentense_to_vec(context)
                x.append(vec)
                y.append(self.one_hot(self.word2int[target], len(self.vocab)))
                c_word += 1
            batch += 1
            if c_word > n_words - windows_size // 2:
                print("word ", c_word)
                c_word = windows_size // 2
            rand = np.random.choice(batch_size, size=batch_size, replace=False)
            x = np.stack(x)
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2],1))
            y = np.stack(y)
            yield x, y




