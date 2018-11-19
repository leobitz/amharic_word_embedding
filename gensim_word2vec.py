import gensim
import logging
import collections
import numpy as np
import random
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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


sentenses = open('data/all.txt', encoding='utf-8').read().split('*')
sentenses = [s.strip().split() for s in sentenses]

params = {
    "embed_size": [[128], [192], [256]],
    "sample": [0.05, 0.001, 0.0005],
    "window": [2, 5, 8],
    "sg": [0, 1],
    "negative": [5, 15, 25],
    "cbow_mean": [0, 1],
}


def multiply(a, b):
    c = []
    for x in a:
        for y in b:
            h = x + [y]
            c.append(h)
    return c


init = params['embed_size']
keys = list(params.keys())[1:]
for key in keys:
    init = multiply(init, params[key])
# init = init#[:11]
experiment_result = []
start = 0
param_file = 'results/gensim_params.txt'
result_file = 'results/gensim_word2vec.txt'
init = [
    [128, 0.0001, 1, 1, 15, 0],
]

for i_params in range(len(init)):
    ini = init[i_params]
    em_size, sample, window, sg, neg, mean = ini
    model = gensim.models.Word2Vec(sentenses,
                                   size=75,
                                   iter=10,
                                #    min_count=1,
                                   negative=15,
                                   sg=0,
                                #    sample=0.00045,
                                   window=5,
                                   cbow_mean=1,
                                   alpha=.05,
                                   workers=10
                                   )
    result = model.accuracy('data/analogies.txt', restrict_vocab=len(model.wv.vocab))
    s = ""   
    for row in result[1]['correct']:
        s += str(row) + "\n"
    open('results/corrects2.txt', mode='w', encoding='utf-8').write(s)
    s = ""   
    for row in result[1]['incorrect']:
        s += str(row) + "\n"
    open('results/incorrect2.txt', mode='w', encoding='utf-8').write(s)
    # analogy_result = calc_analogy_accuracy(result)
    # pick_one_out = round(anomaly(model), 3)
    # analogy_result['semantic_pick'] = pick_one_out

    # s = ''
    # for key in analogy_result:
    #     s += "{0} ".format(analogy_result[key])
    # s += '\n'

    # tex = ' '.join([str(r) for r in init[i_params]]) + '\n'
    # print(i_params, tex, analogy_result)
    # open(result_file, 'a', encoding='utf8').write(s)
    # open(param_file, 'a', encoding='utf8').write(tex)
