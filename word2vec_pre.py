import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops
import time


class Word2VecPre:

    def __init__(self, vocab_size, embed_size=128, batch_size=128, num_sampled=64, unigrams=None):
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
        self._create_optimizer()

    def _create_placeholders(self):
        with tf.name_scope('input_layer'):
            self.train_inputs = tf.placeholder(
                tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(
                tf.int32, shape=[self.batch_size, 1])
            self.reg_labels = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.embedding_size])

    def _create_embedding(self):
        with tf.device('/GPU:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings_layer'):
                init_width = 0.5 / self.embedding_size
                self.embeddings = tf.Variable(
                    tf.random_normal([self.vocab_size, self.embedding_size], -init_width, init_width))
                self.embed = tf.nn.embedding_lookup(
                    self.embeddings, self.train_inputs)
                self.embed = tf.concat([self.embed ,self.reg_labels], axis=1)
            # Construct the variables for the NCE loss
            with tf.name_scope('dense_layer'):
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, self.embedding_size * 2],
                        stddev=1.0 / np.sqrt(self.embedding_size * 2)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

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
            self.loss = tf.reduce_mean(
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
            self.optimizer = tf.train.GradientDescentOptimizer(
                1.0).minimize(self.loss)

    def get_embedding(self):
        return self.embeddings.eval()

    def train_once(self, session, batch_data):

        batch_inputs, batch_labels, batch_reg_labels = batch_data
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.reg_labels: batch_reg_labels}

        loss_val, _ = session.run(
            [self.loss, self.optimizer],
            feed_dict=feed_dict)

        result = {
            "em_loss": loss_val
        }
        return result
