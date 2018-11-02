import tensorflow as tf
import numpy as np
import collections
# from em_data_gen_v2 import *
import time
from word2vec import *
import os


class Trainer:

    def __init__(self, train_name):
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
            training_meta = {}
            for step in range(self.current_epoch, epoches):

                for s in range(steps_per_batch):
                    start_time = time.time()
                    batch_data = next(gen)
                    result = model.train_once(session, batch_data)
                    result['time_taken'] = (time.time() - start_time)
                    self.logger(training_meta, result)

                self.save_model(session, self.model_folder, step, epoches)
                self.save_logs(training_meta)
                log, time_log = self.get_log_text(
                    training_meta, step, epoches, steps_per_batch)
                print(time_log, log)

    def save_model(self, session, folder, step, epoches):
        checkpoint_name = "{0}/model".format(folder)
        self.saver.save(session, checkpoint_name, global_step=step)
        checkpoint = "{0} {1} {2}\n".format(step, epoches, checkpoint_name)
        open(folder + "/record.txt", 'a').write(checkpoint)

    def logger(self, meta: dict, result: dict):
        for res in result.keys():
            if res not in meta:
                meta[res] = []
            meta[res] += [result[res]]

    def get_log_text(self, meta: dict, step, total_step, steps_per_batch):
        log = ""
        for key in meta:
            if key == "time_taken":
                continue
            log += "{1}: {0:.3f} ".format(np.mean(meta[key]), key)

        time_log = {}
        step += 1
        time_log['progress'] = "{0}/{1} {2:.2f}%".format(
            step, total_step, (step * 100 / total_step))
        time_taken = np.mean(meta['time_taken']) * steps_per_batch / 60
        time_log['time'] = "{0:.2f}/{1:.2f}".format(
            time_taken * step, (time_taken * total_step))
        timelog = ""
        for key in time_log.keys():
            timelog += "{0}: {1} ".format(key, time_log[key])
        return log, timelog

    def read_logs(self):
        filename = self.model_folder + "/logs.txt"
        lines = open(filename, encoding='utf8').readlines()
        meta = {}
        for line in lines:
            line = line[:-1].split(' ')
            vals = [float(val) for val in line[1:]]
            meta[line[0]] = vals
        return meta

    def save_logs(self, meta: dict):
        filename = self.model_folder + "/logs.txt"
        s = ""
        for key in meta.keys():
            vals = ' '.join(meta[key])
            s += "{0} {1}\n".format(key, vals)
        open(filename, 'w', encoding='utf8').write(s)
