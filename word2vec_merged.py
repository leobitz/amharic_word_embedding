import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops


class Word2VecMerged:

    def __init__(self, vocab_size, embed_size=128, batch_size=128, num_sampled=64, unigrams=None):
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.unigrams = unigrams
        self.build()
        self.learning_rate = 0.025
        self.word_count = 0
        self.total_words = 0
        self.total_epoches = 0

    def build(self):
        self._create_placeholders()
        self._create_graph()

    def _create_placeholders(self):
        with tf.name_scope('input_layer'):
            self.train_inputs = tf.placeholder(
                tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(
                tf.int32, shape=[self.batch_size])
            self.lr = tf.placeholder(tf.float32, shape=[])

    def _forward(self):
        """Build the graph for the forward pass."""

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / self.embedding_size
        emb = tf.Variable(
            tf.random_uniform(
                [self.vocab_size, self.embedding_size], -init_width, init_width),
            name="emb")
        self.embeddings = emb

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
            tf.zeros([self.vocab_size, self.embedding_size]),
            name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")
        labels = self.train_labels
        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_sampled,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=self.unigrams))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, self.train_inputs)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.matmul(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise lables for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.num_sampled])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def _nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        return nce_loss_tensor

    def _optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train = optimizer.minimize(loss)
        self.optimizer = train

    def _create_graph(self):
        true_logits, sample_logits = self._forward()
        self.loss = self._nce_loss(true_logits, sample_logits)
        self._optimize(self.loss)

    def get_embedding(self):
        return self.embeddings.eval()

    def get_embedding_v2(self, sess):
        embeds = sess.run(self.normalized_embeddings)
        return embeds

    def train_once(self, session, batch_data):
        batch_inputs, batch_labels = batch_data
        lr = max(0.0001, self.learning_rate * (1 -
                                               (self.word_count / (self.total_epoches * self.total_words))))
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels.flatten(),
                     self.lr: lr}

        loss_val, _ = session.run(
            [self.loss, self.optimizer],
            feed_dict=feed_dict)
        self.word_count += len(batch_contexts)
        result = {
            "em_loss": loss_val,
            "lr": lr
        }
        return result
