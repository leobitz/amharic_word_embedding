import gensim, logging
import collections
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def anomaly(model):
    lines = open('data/anomaly.txt', encoding='utf-8').readlines()
    correct = 0
    for line in lines:
        vals = line[:-1].split(' ')
        index = int(vals[-1])
        pred = model.doesnt_match(vals[:4])
        if pred == vals[index]:
            correct += 1
        # print(pred, vals[index])
    return correct * 100 / len(lines)

sentenses = open('data/news.txt', encoding='utf-8').read().split('*')
sentenses = [s.strip().split() for s in sentenses]
model = gensim.models.Word2Vec(sentenses, 
                            size=200, 
                            iter=10, 
                            min_count=1, 
                            sample=0.01,
                            # ns_exponent=1.0
                            )
result = model.accuracy('data/syntax.txt')
result = model.accuracy('data/semantic.txt')

# model1 = gensim.models.Word2Vec(sentenses, size=128, iter=0, min_count=1,)
# model.wv.init_sims()
# model1.wv.init_sims()
# for gindex in range(len(model.wv.index2word)):
#     gword = model.wv.index2word[gindex]
#     model1.wv.vectors_norm[gindex] = model1.wv.vectors_norm[gindex]

# result = model.accuracy('data/syntax.txt')
# result = model.accuracy('data/semantic.txt')

