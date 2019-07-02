from data_handle import *
from gensim_wrapper import *
from utils import *
import gensim
import random
import numpy as np
seed_val = 1000
random.seed(seed_val)
np.random.seed(seed_val)

def parseVec(file, delimiter):
    lines = open(file, encoding='utf8').readlines()
    vocab_size, embed_size = [int(s) for s in lines[0].split()]
    # print(vocab_size, len(word2int))
    vocab_size  = min(len(word2int), vocab_size)
    embeddings = np.zeros((vocab_size + 1, embed_size), dtype=np.float64)
    for i in range(1, vocab_size):
        try:
            line = lines[i][:-1].split(delimiter)
            word = line[0]
            if 'unk' in word:
                continue
            if word in word2int:
                wordvec = np.array([np.float64(j) for j in line[1:] if j != ''])
                embeddings[word2int[word]] = wordvec
        except Exception as e:
            print(e)
    return embeddings

trains = {
     100: 2, 200: 1
}
file = open('train_result', mode='w')
result_dict = {}
for dataId in trains.keys():
    # words = read_file(filename='data/corpus/{0}'.format(dataId))
    # xvocab, xword2int, xint2word = build_vocab(words)
    # newword, word2freq = min_count_threshold(words)
    # del words
    # vocab, word2int, int2word = build_vocab(newword)
    for trainId in range(trains[dataId]):
        # emb = parseVec('data/trains/{0}_100-{1}.vec'.format(dataId, trainId), ' ')
        nr = evaluate3(model_file="data/trains/{0}_100-{1}.vec".format(dataId, trainId),  analogy='data/ar_questions.txt')
        # ur = evaluate3(model_file="data/trains/{0}_100-{1}.vec".format(dataId, trainId), embeddings=normalize(emb), word2int=word2int, analogy='data/ar_questions.txt')
        nsemantic = 0
        nsynatctic = 0
        for nrr in nr:
            if 'total' in nrr:
                continue
            if "gram" in nrr:
                nsynatctic += nr[nrr]/6
            else:
                nsemantic += nr[nrr]/9

        # usemantic = 0
        # usynatctic = 0
        # for nrr in ur:
        #     if 'total' in nrr:
        #         continue
        #     if "gram" in nrr:
        #         usemantic += ur[nrr]/6
        #     else:
        #         usynatctic += ur[nrr]/9
        name = "{0}_{1} ".format(dataId, trainId)
        file.write(name)
        file.write(str([nsynatctic, nsemantic]))
        file.write('\n')
        print(nsynatctic, nsemantic)
    #     del emb
    # del xvocab
    # del xword2int
    # del xint2word
    # del vocab
    # del word2int
    # del int2word
    # del word2freq
    # del newword
file.close()