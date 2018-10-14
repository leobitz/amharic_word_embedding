import numpy as np
import tensorflow as tf


class Word2Vec:

    def __init__(self, vocab_size, embed_size=128, batch_size=128, num_sampled=64):
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.build()

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

    def _create_embedding(self):
        with tf.device('/GPU:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings_layer'):
                self.embeddings = tf.Variable(
                    tf.random_normal([self.vocab_size, self.embedding_size], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(
                    self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('dense_layer'):
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, self.embedding_size],
                        stddev=1.0 / np.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                1.0).minimize(self.loss)

    def get_embedding(self):
        norms = tf.sqrt(tf.reduce_sum(
            tf.square(self.embeddings), 1, keepdims=True))
        normalized_embeddings = self.embeddings / norms
        final_embeddings = normalized_embeddings.eval()
        # final_embeddings = self.embeddings.eval()
        return final_embeddings

    def train_once(self, session, batch_inputs, batch_labels):
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels}

        _, loss_val = session.run(
            [self.optimizer, self.loss],
            feed_dict=feed_dict) 
        return loss_val
    
        
