import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops
import time


def RNN(x, timesteps, units, initial_state):
    x = tf.unstack(x, timesteps, 1)
    grucell = tf.contrib.rnn.GRUCell(num_units=units)
    outputs, states = tf.nn.static_rnn(
        grucell, x, dtype=tf.float32, initial_state=initial_state)
    return outputs, states


class Word2Vec2:

    def __init__(self,
                 vocab_size,
                 n_chars,
                 n_features,
                 embed_size=128,
                 batch_size=128,
                 num_sampled=64,
                 unigrams=None):
        self.n_chars = n_chars
        self.n_features = n_features
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.unigrams = unigrams
        self.build()
        # self._forward()

    def build(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_final_embedding()
        self._create_optimizer()
        self._create_rnn()

    def _create_placeholders(self):
        with tf.name_scope('embedding_inputs'):
            self.train_inputs = tf.placeholder(
                tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(
                tf.int32, shape=[self.batch_size, 1])

        with tf.name_scope('rnn_inputs'):
            self.rnn_inputs = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.n_chars, self.n_features], name="rnn_input")
            self.rnn_targets = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.n_chars, self.n_features], name="rnn_targets")

    def _create_embedding(self):
        with tf.device('/GPU:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings_layer'):
                init_width = 0.5 / self.embedding_size
                self.embeddings = tf.Variable(
                    tf.random_normal([self.vocab_size, self.embedding_size], -init_width, init_width))
                self.embed = tf.nn.embedding_lookup(
                    self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('dense_layer'):
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, self.embedding_size],
                        stddev=1.0 / np.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    def _create_rnn(self):
        with tf.device('/GPU:0'):
            with tf.name_scope('rnn'):
                rnn_w = tf.Variable(
                    tf.truncated_normal(
                        [self.embedding_size, self.n_features],
                        stddev=1.0 / np.sqrt(self.embedding_size)))
                rnn_b = tf.Variable(tf.zeros([self.n_features]))
                initial_state = self.embed
                # rnn = tf.contrib.rnn.GRUCell(num_units=self.embedding_size)
                rnn_outputs, state = RNN(
                    self.rnn_inputs, self.n_chars, self.embedding_size, initial_state)
                # rnn_outputs, state = tf.nn.dynamic_rnn(cell=rnn,
                #                                        inputs=self.rnn_inputs,
                #                                        initial_state=initial_state,
                #                                        time_major=False,
                #                                        dtype=tf.float32)
        with tf.name_scope('rnn_loss'):
            flatOutputs = tf.reshape(
                rnn_outputs, [-1, self.embedding_size])
            rnn_targets = tf.reshape(
                self.rnn_targets, [-1, self.n_features])

            logits = tf.layers.dense(
                flatOutputs, self.n_features, activation=tf.nn.relu)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=rnn_targets)
            loss = tf.reduce_mean(tf.reshape(loss, [self.batch_size, -1]))
            y = tf.argmax(tf.nn.softmax(logits), 1)
            y = tf.reshape(y, [self.batch_size, -1])

            self.rnn_train_step = tf.train.AdamOptimizer(.01).minimize(
                loss)

            correct_pred = tf.equal(
                tf.argmax(logits, 1), tf.argmax(rnn_targets, 1))
            correct_pred = tf.cast(correct_pred, tf.float32)
            self.rnn_accuracy = tf.reduce_mean(correct_pred)
            self.rnn_loss = loss

    def _create_loss(self):
        labels_matrix = tf.cast(self.train_labels, dtype=tf.int64)
        sampled_values = candidate_sampling_ops.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_sampled,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.unigrams
        )
        with tf.name_scope('loss'):
            self.nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size,
                    sampled_values=sampled_values))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(.1).minimize(
                self.nce_loss)

    def _create_final_embedding(self):
        norms = tf.sqrt(tf.reduce_sum(
            tf.square(self.embeddings), 1, keepdims=True))
        self.normalized_embeddings = self.embeddings / norms

    def get_embedding(self):
        norms = tf.sqrt(tf.reduce_sum(
            tf.square(self.embeddings), 1, keepdims=True))
        normalized_embeddings = self.embeddings / norms
        final_embeddings = normalized_embeddings.eval()
        # final_embeddings = self.embeddings.eval()
        return final_embeddings

    def train_once(self, session, batch_inputs, batch_labels,
                   rnn_inputs, rnn_outputs):
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.rnn_inputs: rnn_inputs,
                     self.rnn_targets: rnn_outputs}

        em_time = time.time()
        rnn_loss, rnn_acc = 0, 0
        nce_loss_val, _ = session.run(
            [self.nce_loss, self.optimizer],
            feed_dict=feed_dict)

        em_time = time.time() - em_time
        rnn_time = time.time()

        if np.random.rand() > .5:
            rnn_loss, rnn_acc, _ = session.run(
                [self.rnn_loss, self.rnn_accuracy,
                 self.rnn_train_step],
                feed_dict=feed_dict)

        rnn_time = time.time() - rnn_time

        return nce_loss_val, rnn_loss, rnn_acc, em_time, rnn_time
