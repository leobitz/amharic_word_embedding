from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import keras
import os
import time
from check_point_manager import *
import matplotlib.pyplot as plt
import glob


class TrainingLogger(keras.callbacks.Callback):

    def __init__(self):
        self.total_batches = 0
        self.model_name = ""
        self.total_epoches = 0
        self.save_on = 0
        self.batch_time_start = 0
        self.epoch_start_time = 0
        self.current_epoch = 0
        self.total_time_taken = 0
        self.checkpoint = None
        self.start_time = 0
        self.losses = []

    def on_train_begin(self, logs={}):
        self.model.batch = 10
        self.start_time = time.time()
        try:
            os.stat("model_weights/" + self.model_name)
        except:
            os.mkdir("model_weights/" + self.model_name)
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch += 1
        self.epoch_start_time = time.time()
        return

    def on_epoch_end(self, epoch, logs={}):
        s = '{0}:{1}/{2},'.format("epoch", epoch, self.total_epoches)
        elapsed = time.time() - self.epoch_start_time
        r_epoch = self.total_batches - epoch
        remaining_time = r_epoch * elapsed
        remaining_time = self.pretty_time(remaining_time)
        s += "{0}:{1},".format("elapsed", elapsed)
        s += "{0}:{1},".format('remaining', remaining_time)
        for key in logs:
            val = logs[key]
            info = "{0}:{1},".format(key, val)
            s += info

        self.checkpoint.save(s)
        return

    def on_batch_begin(self, batch, logs={}):
        self.batch_time_start = time.time()
        return

    def on_batch_end(self, batch, logs={}):
        # time_taken = time.time() - self.start_time
        # batch_time = time.time() - self.batch_time_start
        # current_batch = logs.get('batch') + 1
        # if current_batch % self.save_on == 0:
        #     total_iter = self.total_batches * self.total_epoches
        #     till_now = (self.current_epoch - 1) * self.total_batches + batch
        #     rem_iter = total_iter - till_now
        #     remaining_time = rem_iter * batch_time
        #     progress = (self.total_batches * (self.current_epoch - 1) +
        #                 current_batch) * 100 / (self.total_batches * self.total_epoches)
        #     cost = logs.get('loss')
        #     time_taken = self.pretty_time(time_taken)
        #     remaining_time = self.pretty_time(remaining_time)
        #     progress = "Progress: {0:.3f}%, Epoch: {1}/{7}, Batch: {2}/{6}, Loss: {3:.5f}, Taken: {4}, Remaining: {5}\n".format(
        #         progress, self.current_epoch, current_batch, cost, time_taken, remaining_time, self.total_batches, self.total_epoches)
        #     print(progress)
        return

    def pretty_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        return "{0}-{1}-{2}-{3}".format(int(days), int(hrs), int(mins), int(secs))


class TrainingHandler:

    def __init__(self, model, model_name):
        self.n_iterations = 0
        self.current_iter = 0
        self.save_weights_on = 0
        self.model = model
        self.model_name = model_name
        self.latest_weight = None
        self.model_tag = None
        self.time_taken = 0
        self.checkpoint = CheckpointManager()
        try:
            os.stat('model_weights')
        except:
            os.mkdir('model_weights')

    def train(self, training_tag, generator, epoches, batches, save_on, save_model=False, val_gen=None, val_batches=None):
        self.save_weights_on = save_on
        self.model_tag = training_tag
        init_epoch = self.load_last_state()

        history = TrainingLogger()
        history.total_epoches = epoches
        history.save_on = save_on
        history.total_batches = batches
        history.model_name = "{0}_{1}".format(self.model_name, training_tag)
        history.checkpoint = self.checkpoint
        history.checkpoint.prepare(history.model_name)
        history.current_epoch = init_epoch

        if init_epoch == epoches:
            print("Training has ended ")
            return
        per_epoch = batches
        folder = "model_weights/{0}_{1}/".format(
            self.model_name, self.model_tag)
        checkpoint_file = folder + "model_weight-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            checkpoint_file, monitor='loss', verbose=0)
        self.model.fit_generator(generator, steps_per_epoch=per_epoch,
                                 verbose=0, epochs=epoches,
                                 initial_epoch=init_epoch,
                                 callbacks=[checkpoint, history],
                                 shuffle=True,
                                 validation_data=val_gen,
                                 validation_steps=val_batches)
        self.save_history(self.model_tag)

    def load_last_state(self):
        folder = "model_weights/{0}_{1}/*.hdf5".format(
            self.model_name, self.model_tag)
        list_of_files = glob.glob(folder)
        if len(list_of_files) == 0:
            return 0
        last_state = list(sorted(list_of_files))[-1]
        print("Loading State: " + last_state)
        self.model = keras.models.load_model(last_state)
        epoch = int(last_state.split('-')[1])
        return epoch

    def load_best_weight(self, tag):
        best_row, min_cost, iter = self.checkpoint.get_best_state()
        file_name = "model_weights/{0}-{3}/{0}-{1}-{2:.5}.h5".format(
            self.model_name, iter, cost, tag)
        self.model.load_weights(file_name)

    def pretty_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        return "{0}d, {1}h, {2}m, {3:.2f}s".format(days, hrs, mins, secs)

    def clear_old_states(self):
        self.clear_old_states()

    def save_history(self, tag):
        rows = self.checkpoint.get_states()
        filename = "model_weights/{0}_{1}/history.txt".format(
            self.model_name, tag)
        with open(filename, 'a') as file:
            file.writelines([row[1] for row in rows])
