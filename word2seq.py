import numpy as np
import tensorflow as tf

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

    def build(self):
        self._create_placeholders()
        self._create_loss()
        self._create_final_embedding()
        self._create_optimizer()
        self._create_rnn()

    def _create_placeholders(self):
        with tf.name_scope('rnn_inputs'):
            self.rnn_inputs = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.n_chars, self.n_features], name="rnn_input")
            self.rnn_targets = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.n_chars, self.n_features], name="rnn_targets")


    def _create_rnn(self):
        with tf.device('/GPU:0'):
            with tf.name_scope('rnn'):
                rnn_w = tf.Variable(
                    tf.truncated_normal(
                        [self.embedding_size, self.n_features],
                        stddev=1.0 / np.sqrt(self.embedding_size)))
                rnn_b = tf.Variable(tf.zeros([self.n_features]))
                initial_state = self.embed
                rnn = tf.contrib.rnn.GRUCell(num_units=self.embedding_size)
                rnn_outputs, state = tf.nn.dynamic_rnn(cell=rnn,
                                                       inputs=self.rnn_inputs,
                                                       initial_state=initial_state,
                                                       time_major=False,
                                                       dtype=tf.float32)
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

            self.rnn_train_step = tf.train.AdamOptimizer(.001).minimize(
                loss)

            correct_pred = tf.equal(
                tf.argmax(logits, 1), tf.argmax(rnn_targets, 1))
            correct_pred = tf.cast(correct_pred, tf.float32)
            self.rnn_accuracy = tf.reduce_mean(correct_pred)
            self.rnn_loss = loss


    def train_once(self, session, batch_inputs, batch_labels,
                   rnn_inputs, rnn_outputs):
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.rnn_inputs: rnn_inputs,
                     self.rnn_targets: rnn_outputs}

        nce_loss_val, _, rnn_loss, rnn_acc, _ = session.run(
            [self.nce_loss, self.optimizer,
             self.rnn_loss, self.rnn_accuracy,
             self.rnn_train_step],
            feed_dict=feed_dict)
        return nce_loss_val, rnn_loss, rnn_acc
