import numpy as np
import gensim
import logging
import collections
from em_data_gen_v2 import *
import tensorflow as tf
import time
import os
from tensorflow.contrib.tensorboard.plugins import projector

np.random.seed(1000)
filename = "data/news.txt"
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

words, vocab = get_data(filename)
data, word2int, int2word = build_dataset(words)
vocab_size = len(vocab)

# split corpus to sentenses
sentenses = open(filename, encoding='utf-8').read().split('*')
# split each sentense to set of words
# [['word1', 'word2'], ['word2', 'word3']]
sentenses = [s.strip().split() for s in sentenses]
# create gensim word2vec model to get all functionalities like evaluation and nearest word
gensim_model = gensim.models.Word2Vec(
    sentenses, size=embedding_size, iter=0, min_count=1)


def to_gensim_model(gensim_model, word2int, embeddings):
    """Transfers the word embedding learned bu tensorflow to gensim model
    Params:
        gensim_model - un untrained gensim_model
        word2int - dictionary that maps words to int index
        embedding - a new learned embeddings by tensorflow
    """
    for gindex in range(len(gensim_model.wv.index2word)):
        gword = gensim_model.wv.index2word[gindex]
        index = word2int[gword]
        embedding = embeddings[index]
        gensim_model.wv.vectors[gindex] = embedding

def check_last_run(model_folder):
    file = model_folder + "/record.txt"
    if not os.path.exists(file):
        return None
    last_line = open(file).readlines()[-1]
    current_epoch, epoches, model_name = last_line[:-1].split()
    current_epoch = int(current_epoch)
    epoches = int(epoches)
    return current_epoch, epoches, model_name
    

n_words = len(data)
steps_per_batch = n_words // batch_size
epoches = 20
current_epoch = 0

model_name = "first_model"
model_folder = './log/'+ model_name
last_model_name = None
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

last = check_last_run(model_folder)
if last is not None:
    epoches = last[1]
    current_epoch = last[0] 
    last_model_name = last[-1] + "-" + str(current_epoch)

graph = tf.Graph()
with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/GPU:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_normal([vocab_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            print(embed.get_shape())

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocab_size, embedding_size],
                    stddev=1.0 / np.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # Compute the average NCE loss for the batch
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocab_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norms = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norms

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter('./log', session.graph)
    if last_model_name is not None:
        print("Loading model: ", last_model_name)
        saver.restore(session, last_model_name)
        current_epoch += 1
    else:
        init.run()
    # We must initialize all variables before we use them.
    
    print('Initialized', current_epoch, epoches)
    for e in range(current_epoch, epoches):
        average_loss = 0
        start_time = time.time()
        for step in range(steps_per_batch):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips,
                                                        skip_window)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,  # feed dict
                run_metadata=run_metadata)  # visualize in tensorboard
            average_loss += loss_val
        
        # Save the model for checkpoints.
        checkpoint_name = model_folder + "/model-{0}.ckpt".format(e)
        saver.save(session, checkpoint_name, global_step=e)
        checkpoint = "{0} {1} {2}\n".format(e, epoches, checkpoint_name)
        open(model_folder + "/record.txt", 'a').write(checkpoint)

        elapsed_time = time.time() - start_time
        elapsed_mins = elapsed_time / 60
        # Add returned summaries to writer in each step.
        writer.add_summary(summary, e)
        # Add metadata to visualize the graph for the last run.
        writer.add_run_metadata(run_metadata, 'step%d' % e)

        average_loss /= steps_per_batch
        # The average loss is an estimate of the loss over the last 2000 batches.
        ee = e + 1
        log_text = "Progress: {0}/{1} {5:.2f}% Averlage loss: {2:.2f} Time: {3:.2f}/{4:.2f}".format(
            ee, epoches, average_loss, elapsed_mins * ee, (elapsed_mins * epoches), (ee * 100/epoches))
        print(log_text)
        average_loss = 0

    # numpy array of the normalized embeddings
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open('./log/metadata.tsv', 'w', encoding='utf8') as f:
        for i in range(vocab_size):
            f.write(int2word[i] + '\n')

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join('./log', 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
