from em_data_gen import *
from data_gen import *
from model import *


def look2vec(batch, label=False):
    vecs = np.ndarray(shape=(batch_size, n_chars,
                             n_features), dtype=np.float32)
    for i in range(len(batch)):
        b = batch[i]
        w = reverse_dictionary[b]
        if not label:
            w = '&' + w[:-1]
        vec = dg.word2vec2(w)
        vecs[i] = vec
    return vecs


dg = DataGen()
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
np.random.seed(1000)
n_features = len(dg.char2int)
n_chars = 11

graph = tf.Graph()
optimizer, loss, embed, train_inputs, train_labels, normalized_embeddings = embedding_model(graph,
                                                                                            batch_size=batch_size,
                                                                                            vocabulary_size=vocabulary_size,
                                                                                            embedding_size=embedding_size,
                                                                                            num_sampled=num_sampled
                                                                                            )
gru_optimizer, gru_acc, gru_loss, gru_input, gru_targets = syntax_model(graph, embedding_size=embedding_size,
                                                                        embeddings=embed,
                                                                        batch_size=batch_size,
                                                                        n_chars=n_chars,
                                                                        n_features=n_features)
with graph.as_default():
    init = tf.global_variables_initializer()

steps_per_epoch = int(len(data) * num_skips / batch_size)#//1000
print(steps_per_epoch)
num_steps = 2
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(graph=graph, config=config) as session:

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):

        ave_embed_loss, ave_gru_loss, ave_acc = 0, 0, 0
        for batch in range(steps_per_epoch):
            batch_inputs, batch_labels, tt, ty = generate_batch(batch_size, num_skips,
                                                                skip_window, n_chars=n_chars, n_features=n_features)

            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_labels, gru_input: tt, gru_targets: ty}
            _, loss_val, _, gru_loss_val, gru_acc_val = session.run(
                [optimizer, loss, gru_optimizer, gru_loss, gru_acc],
                feed_dict=feed_dict)
            ave_gru_loss += gru_loss_val
            ave_acc += gru_acc_val
            ave_embed_loss += ave_embed_loss

        ave_acc = ave_acc / steps_per_epoch
        ave_embed_loss = ave_embed_loss / steps_per_epoch
        ave_gru_loss = ave_gru_loss / steps_per_epoch
        step_log = "Epoch: {0} Embed Loss: {1} GRU Loss: {2} GRU Acc: {3}".format(
            step, ave_embed_loss, ave_gru_loss, ave_acc)
        print(step_log)

    final_embeddings = normalized_embeddings.eval()
