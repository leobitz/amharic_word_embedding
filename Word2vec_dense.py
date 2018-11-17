import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops
import time


class Word2VecDense:

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
            self.context_word = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.embedding_size])
            self.target_word = tf.placeholder(
                tf.float32, shape=[self.batch_size, self.embedding_size])
            self.labels = tf.placeholder(
                tf.int32, shape=[self.batch_size, 1])

    def _create_embedding(self):
        with tf.device('/GPU:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings_layer'):
                init_width = 0.5 / self.embedding_size
                self.context_layer = tf.layers.dense(
                    self.context_word, self.embedding_size)
                self.target_layer = tf.layers.dense(
                    self.target_word, self.embedding_size, activation=tf.nn.relu)
                x = tf.concat([self.context_layer, self.target_layer], axis=1)

            # Construct the variables for the NCE loss
            with tf.name_scope('final_layer'):
                self.nce_w = tf.Variable(
                    tf.truncated_normal(
                        [self.embedding_size, self.vocab_size],
                        stddev=1.0 / np.sqrt(self.embedding_size)))
                self.nce_b = tf.Variable(tf.zeros([self.vocab_size]))

            with tf.name_scope('output_layer'):
                self.outputs = tf.matmul(self.context_layer, self.nce_w) + self.nce_b

    def _create_loss(self):
        labels_matrix = tf.cast(self.labels, dtype=tf.int64)
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
                tf.nn.sampled_softmax_loss(
                    weights=self.nce_w,
                    biases=self.nce_b,
                    labels=self.labels,
                    inputs=self.outputs,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size,
                    num_true=1,
                    sampled_values=sampled_values))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                0.1).minimize(self.loss)

    def get_embedding(self, session, seq_embeddings):
        feed_dict = {self.context_word: seq_embeddings,
                     self.target_word: seq_embeddings}
        contexts, targets = session.run(
            [self.context_layer, self.target_word], feed_dict=feed_dict)
        return contexts, targets

    def train_once(self, session, batch_data):

        batch_contexts, batch_targets, batch_labels = batch_data
        feed_dict = {self.context_word: batch_contexts,
                     self.target_word: batch_targets,
                     self.labels: batch_labels}

        loss_val, _ = session.run(
            [self.loss, self.optimizer],
            feed_dict=feed_dict)

        result = {
            "em_loss": loss_val
        }
        return result
