from tf_tester import Tester
from tf_trainer import *
from word2vec import *
from gensim_wrapper import *
from utils import Utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '9'

batch_size = 1000
embedding_size = 128
# skip_window = 1
# num_skips = 2
# num_sampled = 64

model_name = "log/full_200/model-19"
name = "test"
tester = Tester()
gensim_model = GensimWrapper(embedding_size, 0, log=True)
vocab_size = tester.vocab_size

graph = tf.Graph()
with graph.as_default():
    wv_model = Word2Vec(vocab_size=vocab_size,
                        embed_size=embedding_size,
                        batch_size=batch_size)
tester.evaluate(graph, wv_model, gensim_model, model_name)

utils = Utils(tester.word2int, tester.embeddings)
print(utils.sorted_sim('አቶ'))
print(utils.sorted_sim('ነው'))
print(utils.sorted_sim('ኢትዮጵያ'))
