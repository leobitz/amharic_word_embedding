import tensorflow as tf
import numpy as np


def syntax_model(graph, embeddings, embedding_size, batch_size, n_chars, n_features):
    with graph.as_default():

        with tf.name_scope('gru_inputs'):

            gru_inputs = tf.placeholder(tf.float32,
                                        shape=[batch_size,
                                               n_chars, n_features],
                                        name="gru_inputs")
            gru_targets = tf.placeholder(tf.float32,
                                         shape=[batch_size,
                                                n_chars, n_features],
                                         name="gru_targets")

        with tf.device('/gpu:0'):
            with tf.name_scope('gru'):
                gru_w = tf.Variable(
                    tf.truncated_normal(
                        [embedding_size, n_features],
                        stddev=1.0 / np.sqrt(embedding_size)))
                gru_b = tf.Variable(tf.zeros([n_features]))

                initial_state = embeddings
                gru_cell = tf.contrib.rnn.GRUCell(num_units=embedding_size)
                gru_outputs, gru_state = tf.nn.dynamic_rnn(cell=gru_cell,
                                                           inputs=gru_inputs,
                                                           initial_state=initial_state,
                                                           time_major=False,
                                                           dtype=tf.float32)

                flatOutputs = tf.reshape(gru_outputs, [-1, embedding_size])
                gru_label = tf.reshape(gru_targets, [-1, n_features])
                gru_logits = tf.layers.dense(
                    flatOutputs, n_features, activation=tf.nn.relu)
        with tf.name_scope("gru_loss"):
            gru_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=gru_logits, labels=gru_label)
            gru_loss = tf.reshape(gru_loss, [batch_size, -1])
            gru_loss = tf.reduce_mean(gru_loss)
            gru_softmax = tf.nn.softmax(gru_logits)
            gru_y = tf.argmax(gru_softmax, 1)
            gru_y = tf.reshape(gru_y, [batch_size, -1])

            # accuracy calculate
            gru_prediction = tf.equal(
                tf.argmax(gru_logits, 1), tf.argmax(gru_label, 1))
            gru_prediction = tf.cast(gru_prediction, tf.float32)
            gru_acc = tf.reduce_mean(gru_prediction)
            # train step for task 2
            gru_task_train_step = tf.train.AdamOptimizer(.001).minimize(gru_loss)
        return gru_task_train_step, gru_acc, gru_loss, gru_inputs, gru_targets


def embedding_model(graph, batch_size, vocabulary_size, embedding_size, num_sampled):
    with graph.as_default():

        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('nce_weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, embedding_size],
                        stddev=1.0 / np.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # NCE loss for embedding task
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('embedding_optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        # valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
        #                                         valid_dataset)
        # similarity = tf.matmul(
        #     valid_embeddings, normalized_embeddings, transpose_b=True)
        # # Merge all summaries.
        # merged = tf.summary.merge_all()

        # # Add variable initializer.

        # # Create a saver.
        # saver = tf.train.Saver()
        return optimizer, loss, embed, train_inputs, train_labels, normalized_embeddings



