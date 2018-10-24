import tensorflow as tf
import numpy as np
import collections
import time
from word2vec_rnn import *
import os


class Logger:

    def __init__(self, multi_task=False):
        self.nce_losses = []
        if multi_task:
            self.rnn_losses = []
            self.rnn_acc = []

    def log(self, result):
        if type(result) == tuple:
            self.nce_losses.append(result[0])
            self.rnn_losses.append(result[1])
            self.rnn_acc.append(result[2])
        else:
            self.nce_losses.append(result)

    def get_log(self, multi_task=False):
        if multi_task:
            nce_loss = sum(self.nce_losses) / len(self.nce_losses)
            rnn_loss = sum(self.rnn_losses) / len(self.nce_losses)
            rnn_acc = sum(self.rnn_acc) / len(self.nce_losses)
            log = "NCE Loss: {0:.2f} RNN Loss: {1:.2f} RNN Acc: {2:.2f}".format(
                nce_loss, rnn_loss, rnn_acc)
        else:
            nce_loss = sum(self.nce_losses) / len(self.nce_losses)
            log = "NCE Loss: {0:.2f}".format(nce_loss)
        return log


class RnnTrainer:

    def __init__(self, train_name, batch_size=128, skip_window=1):
        self.batch_size = batch_size
        self.skip_window = skip_window
        self.train_name = train_name
        self.current_epoch = 0
        self._prepare_folder()
        self._prepare_last_model()

    def _prepare_folder(self):
        self.model_folder = './log/' + self.train_name
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

    def train(self, graph, model, gen, steps_per_batch, embed_size, epoches=10):
        with graph.as_default():
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:

            if self.last_model_name is not None:
                print("Loading model: ", self.last_model_name)
                self.saver.restore(session, self.last_model_name)
                self.current_epoch += 1
            else:
                init.run()
            for step in range(self.current_epoch, epoches):
                start_time = time.time()
                timing = 0
                times = []
                time_of_s = time.time()
                em_time = 0
                rnn_time = 0
                eloss, rloss, racc = [], [], []
                for s in range(steps_per_batch):
                    # data_start_time = time.time()
                    batch_inputs, batch_labels, rnn_inputs, rnn_labels = next(
                        gen)
                    # data_end_time = time.time() - data_start_time
                    # run_start = time.time()
                    result = model.train_once(session,
                                              batch_inputs, batch_labels,
                                              rnn_inputs, rnn_labels)
                    eloss.append(result[0])
                    rloss.append(result[1])
                    racc.append(result[2])
                    # run_end = time.time() - run_start
                    # times.append([data_end_time, run_end])
                    # timing += 1
                    # em_time += result[-2]
                    # rnn_time += result[-1]
                    # if timing == 1000:
                    #     timing = 0
                    #     times = np.array(times)
                    #     times_ave = times.sum(axis=0)
                    #     times = []
                    #     print(times_ave)
                    if s % 1000 == 0:
                        # time_of_s = (time.time() - time_of_s)
                        print("{0}/{1} {2:.2f} {3:.2f} {4:.2f}".format(s,
                                                                       steps_per_batch, np.mean(eloss), np.mean(rloss), np.mean(racc)))
                        # time_of_s = time.time()
                        # em_time = 0
                        # rnn_time = 0

                self.save_model(session, self.model_folder, step, epoches)
                elapsed_mins = (time.time() - start_time) / 60
                ee = step + 1

                log_text = "Progress: {0}/{1} {5:.2f}% Averlage loss: {2:.2f} Time: {3:.2f}/{4:.2f}".format(
                    ee, epoches, result[0], elapsed_mins * ee, (elapsed_mins * epoches), (ee * 100 / epoches))
                print(log_text)

    def save_model(self, session, folder, step, epoches):
        checkpoint_name = "{0}/model".format(folder)
        self.saver.save(session, checkpoint_name, global_step=step)
        checkpoint = "{0} {1} {2}\n".format(step, epoches, checkpoint_name)
        open(folder + "/record.txt", 'a').write(checkpoint)

    def handler_log(self, result):
        if type(result) == float:
            nce_loss = result[0]
            rnn_loss = result[1]
            rnn_acc = result[2]
            return log
        else:
            return "NCE Loss: {0:.2f}".format(result)
