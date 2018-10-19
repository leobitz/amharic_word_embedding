import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops


class Word2Vec:

    def __init__(self, vocab_size, wordfreq, embed_size=128, batch_size=128, num_sampled=64, unigrams=None):
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.unigrams = unigrams
        self.word_freq = wordfreq
        self.build()
        # self._forward()

    def build(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_final_embedding()
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

    def get_embedding_v2(self, sess):
        embeds = sess.run(self.normalized_embeddings)
        # print(embeds.get_size())
        return embeds

    def train_once(self, session, batch_inputs, batch_labels):
        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels}

        loss_val, _ = session.run(
            [self.loss, self.optimizer],
            feed_dict=feed_dict)
        return loss_val

    # def _forward(self):
    #     """Build the graph for the forward pass."""
    #     opts = self
    #     examples = self.train_inputs
    #     labels = self.train_labels
    #     word_freq = self.word_freq
    #     # Declare all variables we need.
    #     # Embedding: [vocab_size, emb_dim]
    #     init_width = 0.5 / self.embedding_size
    #     emb = tf.Variable(
    #         tf.random_uniform(
    #             [opts.vocab_size, opts.embedding_size], -init_width, init_width),
    #         name="emb")
    #     self.embeddings = emb

    #     # Softmax weight: [vocab_size, emb_dim]. Transposed.
    #     sm_w_t = tf.Variable(
    #         tf.zeros([opts.vocab_size, opts.embedding_size]),
    #         name="sm_w_t")

    #     # Softmax bias: [emb_dim].
    #     sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

    #     # Global step: scalar, i.e., shape [].
    #     self.global_step = tf.Variable(0, name="global_step")

    #     # Nodes to compute the nce loss w/ candidate sampling.
    #     labels_matrix = tf.cast(labels, dtype=tf.int64)

    #     # Negative sampling.
    #     sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
    #         true_classes=labels_matrix,
    #         num_true=1,
    #         num_sampled=opts.num_sampled,
    #         unique=True,
    #         range_max=opts.vocab_size,
    #         distortion=0.75,
    #         unigrams=list(word_freq.values()))

    #     # Embeddings for examples: [batch_size, emb_dim]
    #     example_emb=tf.nn.embedding_lookup(emb, examples)

    #     # Weights for labels: [batch_size, emb_dim]
    #     true_w=tf.nn.embedding_lookup(sm_w_t, labels)
    #     # Biases for labels: [batch_size, 1]
    #     true_b=tf.nn.embedding_lookup(sm_b, labels)

    #     # Weights for sampled ids: [num_sampled, emb_dim]
    #     sampled_w=tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    #     # Biases for sampled ids: [num_sampled, 1]
    #     sampled_b=tf.nn.embedding_lookup(sm_b, sampled_ids)

    #     # True logits: [batch_size, 1]
    #     true_logits=tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

    #     # Sampled logits: [batch_size, num_sampled]
    #     # We replicate sampled noise lables for all examples in the batch
    #     # using the matmul.
    #     sampled_b_vec=tf.reshape(sampled_b, [opts.num_sampled])
    #     sampled_logits=tf.matmul(example_emb,
    #                                sampled_w,
    #                                transpose_b=True) + sampled_b_vec
    #     return true_logits, sampled_logits

    # def _nce_loss(self, true_logits, sampled_logits):
    #     """Build the graph for the NCE loss."""

    #     # cross-entropy(logits, labels)
    #     opts=self
    #     true_xent=tf.nn.sigmoid_cross_entropy_with_logits(
    #         true_logits, tf.ones_like(true_logits))
    #     sampled_xent=tf.nn.sigmoid_cross_entropy_with_logits(
    #         sampled_logits, tf.zeros_like(sampled_logits))

    #     # NCE-loss is the sum of the true and noise (sampled words)
    #     # contributions, averaged over the batch.
    #     nce_loss_tensor=(tf.reduce_sum(true_xent) +
    #                        tf.reduce_sum(sampled_xent)) / opts.batch_size
    #     return nce_loss_tensor

    # def _optimize(self, loss):
    #     """Build the graph to optimize the loss function."""

    #     # Optimizer nodes.
    #     # Linear learning rate decay.
    #     opts=self
    #     words_to_train=float(opts.words_per_epoch * opts.epochs_to_train)
    #     lr=opts.learning_rate * tf.maximum(
    #         0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
    #     self._lr=lr
    #     optimizer=tf.train.GradientDescentOptimizer(lr)
    #     train=optimizer.minimize(loss,
    #                                global_step=self.global_step,
    #                                gate_gradients=optimizer.GATE_NONE)
    #     self._train=train
