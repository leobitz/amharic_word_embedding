import gensim, logging
import collections

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentenses = open('data/all.txt', encoding='utf-8').read().split('*')
sentenses = [s.strip().split() for s in sentenses]
model = gensim.models.Word2Vec(sentenses, size=200, iter=20, min_count=1)
model.accuracy('data/syntax.txt')
