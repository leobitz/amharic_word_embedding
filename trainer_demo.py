from tf_trainer import *
from word2vec import *
from gensim_wrapper import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 1000
embedding_size = 128
skip_window = 5
num_skips = 2
trainer = Trainer(train_name="full_200", 
                batch_size=batch_size,
                  skip_window=skip_window,
                  max_words=-1)
vocab_size = trainer.vocab_size

trainer.train(embedding_size, epoches=40)
