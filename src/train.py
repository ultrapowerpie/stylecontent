from LSTM import LSTM
import numpy as np
import tensorflow as tf

'''Train and sample from our model'''

# data I/O
# should be simple plain text file
corpus = open('../data/output/dr_seuss_phones.txt', 'r').read().split(" ")
data = corpus#[:int(len(corpus)*0.9)]
syllabs = list(set(data))
data_size, vocab_size = len(data), len(syllabs)
print 'data has %d syllables, %d unique.' % (data_size, vocab_size)
syllab_to_ix = { s:i for i,s in enumerate(syllabs) }
ix_to_syllab = { i:s for i,s in enumerate(syllabs) }


# hyperparameters
embedding_size = 32 # size of embedding
hidden_size = 256 # size of hidden layers of neurons
seq_length = 50 # number of steps to unroll the LSTM for
learning_rate = 1e-2
decay_steps = 500
decay_factor = 0.9
sample_len = 500

batch_size = 128

n_train_steps = 1

# model parameters
lstm = LSTM(batch_size, embedding_size, hidden_size, vocab_size,
          seq_length, learning_rate, decay_steps, decay_factor,
          sample_len, GPU=True)

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

for n in range(n_train_steps):

    # prepare inputs
    inputs = np.empty([batch_size, seq_length])
    targets = np.empty([batch_size, seq_length])

    for i in range(batch_size):
        # randomly index into the data for each example in batch
        random_index = int(np.random.rand() * (data_size - seq_length - 1))
        inputs[i, :] = [syllab_to_ix[ch] for ch in data[random_index:random_index+seq_length]]
        targets[i, :] = [syllab_to_ix[ch] for ch in data[random_index+1:random_index+seq_length+1]]

    loss, lr = lstm.run_train(inputs, targets)

    # print progress
    if n % 100 == 0:
        print 'iter %d, loss: %f, learning rate: %f' % (n, loss, lr)

    # sample from the model now and then
    if n % 1000 == 0:
        sample_ix = lstm.run_sample(sample_len, inputs[0, 0], 1.0)
        txt = ' '.join(ix_to_syllab[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )

# cross-entropy on test set
test = corpus[int(len(corpus)*0.9):]
primer = data[-1000:]
lstm.run_test([syllab_to_ix[ch] for ch in test], [syllab_to_ix[ch] for ch in primer])
