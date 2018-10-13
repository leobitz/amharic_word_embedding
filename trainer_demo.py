from tf_trainer import *
from word2vec import *
from gensim_wrapper import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 2
num_sampled = 1000

trainer = Trainer(train_name="full_200", 
                batch_size=batch_size,
                  skip_window=skip_window,
                  max_words=-1)
vocab_size = trainer.vocab_size

gensim_model = GensimWrapper(embedding_size, iter=0)

graph = tf.Graph()
with graph.as_default():
    wv_model = Word2Vec(vocab_size=vocab_size,
                        embed_size=embedding_size,
                        batch_size=batch_size)
trainer.train(graph, wv_model, gensim_model, epoches=10)
