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
                            iter=0, 
                            min_count=1, 
                            sample=1.0,
                            # ns_exponent=1.0
                            )
result = model.accuracy('data/syntax.txt')
result = model.accuracy('data/semantic.txt')
print(result[0]['correct'])
# model1 = gensim.models.Word2Vec(sentenses, size=128, iter=0, min_count=1,)
# model.wv.init_sims()
# model1.wv.init_sims()
# for gindex in range(len(model.wv.index2word)):
#     gword = model.wv.index2word[gindex]
#     model1.wv.vectors_norm[gindex] = model1.wv.vectors_norm[gindex]

# result = model.accuracy('data/syntax.txt')
# result = model.accuracy('data/semantic.txt')
# print(result)
# params = {
#     "embed_size": [100, 125, 150, 175, 200, 225, 250, 275],
#     "sample": np.linspace(0.01, 0.0001, 10),
#     "window": [1, 3, 5, 7, 9, 11],
#     "sg": [0, 1],
#     "negative": [5, 15, 25, 35, 45, 55, 65, 75, 85],
#     "cbow_mean": [0, 1],
# }

# grid = 1
# random = {}
# for key in params:
#     val = params[key]
#     if len(val)  < 3:
#         random[key] = val
#     else:
#         random[key] = np.random.choice(val, min(len(val), 3), replace=False)
#     print(random[key])




