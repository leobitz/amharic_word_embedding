from tf_tester import Tester
from tf_trainer import *
from word2vec import *
from gensim_wrapper import *
from utils import Utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 10000
embedding_size = 128
# skip_window = 1
# num_skips = 2
# num_sampled = 64

model_name = "log/full_200/model-39"
name = "test"
tester = Tester()
gensim_model = GensimWrapper(embedding_size, 0, log=True)
vocab_size = tester.vocab_size


tester.evaluate(gensim_model, model_name, batch_size, embedding_size)

utils = Utils(tester.word2int, tester.embeddings)
print(utils.sorted_sim('አቶ'))
print(utils.sorted_sim('ነው'))
print(utils.sorted_sim('ኢትዮጵያ'))
