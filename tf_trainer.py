import tensorflow as tf
import numpy as np
import collections
from em_data_gen_v2 import *
import time
import os


class Trainer:

    def __init__(self, train_name, batch_size=128, skip_window=1, max_words=-1):
        self.batch_size = batch_size
        self.skip_window = skip_window 
        self.num_skips = 2
        self.train_name = train_name
        self.current_epoch = 0
        self._prepare_folder()
        self._prepare_data(max_words)
        self._prepare_last_model()
    
    def _prepare_data(self, max_words):
        filename = "data/news.txt"
        self.words, self.vocab = get_data(filename, max_words=max_words)
        self.data, self.word2int, self.int2word = build_dataset(self.words)
        self.vocab_size = len(self.vocab)
        print("Words: {0} Vocab: {1}".format(len(self.words), self.vocab_size))
    
    def _prepare_folder(self):
        self.model_folder = './log/'+ self.train_name
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
    
    def _prepare_last_model(self):
        self.last_model_name = None
        last = self.get_last_run()
        if last is not None:
            self.current_epoch = last[0]
            self.last_model_name = last[-1] 

    def get_last_run(self):
        file = self.model_folder + "/record.txt"
        if not os.path.exists(file):
            return None
        last_line = open(file).readlines()[-1]
        current_epoch, epoches, model_name = last_line[:-1].split()
        current_epoch = int(current_epoch)
        epoches = int(epoches)
        return current_epoch, epoches, model_name + '-' + str(current_epoch)

    def train(self, graph, w2v_model, gensim_model, epoches=10):
        with graph.as_default():
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        steps_per_batch = len(self.data) // self.batch_size

        with tf.Session(graph=graph) as session:

            if self.last_model_name is not None:
                print("Loading model: ", self.last_model_name)
                self.saver.restore(session, self.last_model_name)
                self.current_epoch += 1
            else:
                init.run()
            gen = generate_batch_v2(self.batch_size, self.skip_window)
            for step in range(self.current_epoch, epoches):
                average_loss = 0
                start_time = time.time()
                for s in range(steps_per_batch):
                    batch_inputs, batch_labels = next(gen)

                    average_loss += w2v_model.train_once(session,
                                                     batch_inputs, batch_labels)
                self.save_model(session, self.model_folder, step, epoches)
                
                elapsed_time = time.time() - start_time
                elapsed_mins = elapsed_time / 60

                average_loss /= steps_per_batch
                ee = step + 1
                log_text = "Progress: {0}/{1} {5:.2f}% Averlage loss: {2:.2f} Time: {3:.2f}/{4:.2f}".format(
                    ee, epoches, average_loss, elapsed_mins * ee, (elapsed_mins * epoches), (ee * 100/epoches))
                print(log_text)
                average_loss = 0

    def save_model(self, session, folder, step, epoches):
        checkpoint_name = "{0}/model".format(folder)
        self.saver.save(session, checkpoint_name, global_step=step)
        checkpoint = "{0} {1} {2}\n".format(step, epoches, checkpoint_name)
        open(folder + "/record.txt", 'a').write(checkpoint)