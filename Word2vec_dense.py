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
                shape=[self.batch_size], dtype=tf.int32)
            self.target_word = tf.placeholder(
                shape=[self.batch_size, 1], dtype=tf.int32)
            self.seq_encoding = tf.placeholder(
                shape=[self.batch_size, 50], dtype=tf.float32)
            # print(self.context_word.get_shape())
    def _create_embedding(self):
        with tf.device('/GPU:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings_layer'):
                init_width = 0.5 / self.embedding_size
                self.embeddings = tf.Variable(
                    tf.random_normal([self.vocab_size, self.embedding_size],
                                     -init_width, init_width, dtype=np.float32), dtype=tf.float32)
                self.embedx = tf.nn.embedding_lookup(
                    self.embeddings, self.context_word)
                x = tf.concat([self.embedx, self.seq_encoding], axis=1)
                x = tf.layers.dense(x, self.embedding_size, activation=tf.nn.relu)
                self.embed = tf.layers.dense(x, self.embedding_size)

            # Construct the variables for the NCE loss
            with tf.name_scope('final_layer'):
                self.nce_w = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, self.embedding_size],
                        stddev=1.0 / np.sqrt(self.embedding_size), dtype=tf.float32), dtype=tf.float32)
                self.nce_b = tf.Variable(
                    tf.zeros([self.vocab_size], dtype=tf.float32), dtype=tf.float32)

    def _create_loss(self):
        labels_matrix = tf.cast(self.target_word, dtype=tf.int64)
        sampled_values = candidate_sampling_ops.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_sampled,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.unigrams
        )
        print(self.embed.get_shape(), labels_matrix.get_shape(), self.num_sampled)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=self.nce_w,
                    biases=self.nce_b,
                    labels=self.target_word,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size,
                    num_true=1,
                    sampled_values=sampled_values))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                0.1).minimize(self.loss)

    def get_embedding(self):
        return self.embeddings.eval()
    # def get_embedding(self, session, seq_embeddings):
    #     feed_dict = {self.context_word: seq_embeddings,
    #                  self.target_word: seq_embeddings}
    #     contexts, targets = session.run(
    #         [self.context_layer, self.target_word], feed_dict=feed_dict)
    #     return contexts, targets

    def get_embed_2(self, session, seq_enc, inputs):
        feed_dict = {self.context_word: inputs,
                     self.seq_encoding: seq_enc}
        embeds = session.run(self.embed, feed_dict=feed_dict)
        return embeds

    def train_once(self, session, batch_data):

        batch_contexts, batch_targets, batch_seq_encoding = batch_data
        feed_dict = {self.context_word: batch_contexts,
                     self.target_word: batch_targets,
                     self.seq_encoding: batch_seq_encoding}

        loss_val, _ = session.run(
            [self.loss, self.optimizer],
            feed_dict=feed_dict)

        result = {
            "em_loss": loss_val
        }
        return result
