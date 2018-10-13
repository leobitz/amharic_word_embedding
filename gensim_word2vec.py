import gensim, logging
import collections

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

sentenses = open('data/all.txt', encoding='utf-8').read().split('*')
sentenses = [s.strip().split() for s in sentenses]
model = gensim.models.Word2Vec(sentenses, size=200, iter=1, min_count=1)
result = model.accuracy('data/syntax.txt')

