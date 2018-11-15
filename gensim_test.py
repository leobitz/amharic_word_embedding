import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import logging
import collections
import numpy as np
import random
import os
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
random.seed(1000)
np.random.seed(1000)


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


def calc_analogy_accuracy(result):
    results = {}
    for val in result:
        name = val['section']
        n_correct = len(val['correct'])
        n_incorrect = len(val['incorrect'])
        acc = round(n_correct * 100 / (n_correct + n_incorrect), 3)
        results[name] = acc
    return results


sentenses = open('data/news.txt', encoding='utf-8').read().split('*')
sentenses = [s.strip().split() for s in sentenses]
params = [em_size, neg, sg,sample,window,mean] = 128, 5, 0, 0.0001, 3, 0
model = gensim.models.Word2Vec(sentenses,
                               size=em_size,
                               iter=20,
                               min_count=1,
                               negative=neg,
                               sg=sg,
                               sample=sample,
                               window=window,
                               cbow_mean=mean,
                               workers=10,
                               seed=1000
                               )
analogy_result = calc_analogy_accuracy(
    model.accuracy('data/analogies.txt'))
pick_one_out = round(anomaly(model), 3)
analogy_result['semantic_pick'] = pick_one_out
print(params)
print(analogy_result)
