import numpy as np
import tensorflow as tf

class LSTM(object):

    def __init__(self, batch_size, embedding_size, hidden_size, vocab_size, seq_length,
                 learning_rate, decay_steps, decay_factor, sample_len, GPU=False):
        ''' Set the hyperparameters and define the computation graph.
        '''

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size # number of syllabs in vocab
        self.seq_length = seq_length # number of steps to unroll the LSTM for
        self.initial_learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.sample_len = sample_len

        # this var keeps track of the train steps within the LSTM
        self.global_step = tf.Variable(0, trainable=False)

        ''' create vars and graph '''

        if GPU:
            with tf.device("/gpu:0"):
                self._init_params()
                self._build_graph()
        else:
            with tf.device("/cpu:0"):
                self._init_params()
                self._build_graph()

    def _init_params(self):
        '''Create the model parameters'''

        # Learn an embedding for each syllable jointly with the other model params
        self.embedding = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size],
                                                      mean=0, stddev=0.2))
        self.Uf = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size],
                                       mean=0, stddev=0.2))
        self.Ui = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size],
                                       mean=0, stddev=0.2))
        self.Uo = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size],
                                       mean=0, stddev=0.2))
        self.Uc = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size],
                                               mean=0, stddev=0.2))
        self.Wf = tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size],
                                               mean=0, stddev=0.2))
        self.Wi = tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size],
                                               mean=0, stddev=0.2))
        self.Wo = tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size],
                                               mean=0, stddev=0.2))
        self.Wc = tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size],
                                               mean=0, stddev=0.2))
        self.V = tf.Variable(tf.random_normal([self.hidden_size, self.vocab_size],
                                               mean=0, stddev=0.2))

        self.bf = tf.Variable(tf.zeros([1, self.hidden_size]))
        self.bi = tf.Variable(tf.zeros([1, self.hidden_size]))
        self.bo = tf.Variable(tf.zeros([1, self.hidden_size]))
        self.bc = tf.Variable(tf.zeros([1, self.hidden_size]))
        self.by = tf.Variable(tf.zeros([1, self.vocab_size]))

    def _lstm_step(self, x, h, c):
        '''Performs LSTM computation for one timestep:
        takes a previous x and h, and computes the next x and h.
        '''

        f = tf.nn.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(h, self.Uf) + self.bf)
        i = tf.nn.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(h, self.Ui) + self.bi)
        o = tf.nn.sigmoid(tf.matmul(x, self.Wo) + tf.matmul(h, self.Uo) + self.bo)
        uc = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(h, self.Uc) + self.bc)
        c = tf.multiply(f, c)+tf.multiply(i, uc)
        h = tf.multiply(o, tf.nn.tanh(c))
        y = tf.matmul(h, self.V)+self.by

        return y, h, c


    def _forward(self, inputs):
        '''Performs the forward pass for all timesteps in a sequence.
        '''
        # Create list to hold y
        y = [_ for _ in range(self.seq_length)]

        # Create zero-d initial hidden state
        h = tf.zeros([self.batch_size, self.hidden_size])
        c = tf.zeros([self.batch_size, self.hidden_size])


        for t in range(self.seq_length):
            x = tf.nn.embedding_lookup(self.embedding, inputs[:, t])
            y[t], h, c = self._lstm_step(x, h, c)

        return y


    def _sample_one(self, input_syllab, input_hidden, cell_state, temperature):
        '''Sample the single next syllable in a sequence.'''

        # We expand dims because tf expects a batch
        syllab = tf.expand_dims(input_syllab, 0)

        # Get the embedding for the input syllable
        x = tf.nn.embedding_lookup(self.embedding, syllab)

        # Take a single lstm step
        y, h, c = self._lstm_step(x, input_hidden, cell_state)

        # Dividing the unnormalized probabilities by the temperature before
        # tf.multinomial is equivalent to adding temperature to a softmax
        # before sampling
        y_temperature = y / temperature

        # We use tf.squeeze to remove the unnecessary [batch, num_samples] dims
        # We do not manually softmax - tf.multinomial softmaxes the tensor we pass it
        next_sample = tf.squeeze(tf.multinomial(y_temperature, 1))

        return next_sample, h, c, y


    def _build_graph(self):
        '''Build the computation graphs for training and sampling.'''


        '''Sampling and test graph'''
        self.sample_input_syllab = tf.placeholder(dtype=tf.int32, shape=[])
        self.sample_cell_state = tf.placeholder(dtype=tf.float32, shape=[1, self.hidden_size])
        self.sample_input_hidden = tf.placeholder(dtype=tf.float32, shape=[1, self.hidden_size])

        self.test_syllab = tf.placeholder(dtype=tf.int32, shape=[])

        self.temperature = tf.placeholder_with_default(1.0, [])

        self.next_sample, self.next_hidden, self.next_cell, self.next_predictions = self._sample_one(
            self.sample_input_syllab, self.sample_input_hidden, self.sample_cell_state, self.temperature)

        self.next_softmax_predictions = tf.nn.softmax(self.next_predictions)

        self.test_syllab_prob = tf.reduce_sum(self.next_softmax_predictions * tf.one_hot(
            tf.expand_dims(self.test_syllab, axis=0), depth=self.vocab_size))

        # Get cross entropy in base 2
        # log_2 (x) =  log_e (x) / log_e(2)
        self.binary_xentropy = - tf.log(self.test_syllab_prob) / tf.log(2.0)


        '''Training graph'''
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length])
        self.predictions = self._forward(self.inputs)

        cost_per_timestep_per_example = [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.predictions[t],
                    labels=self.targets[:, t])
                for t in range(self.seq_length)
        ]

        # Use reduce_mean rather than reduce_sum over the examples in batch so that
        # we don't need to change the learning rate when we change the batch size.
        cost_per_timestep = [tf.reduce_mean(cost) for cost in cost_per_timestep_per_example]

        # Use reduce_mean here too so we don't need to change the learning rate when
        # we change number of timesteps.
        self.cost = tf.reduce_mean(cost_per_timestep)

        # Decay the learning rate according to a schedule.
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                        self.global_step,
                                                        self.decay_steps,
                                                        self.decay_factor)

        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(
            self.cost, global_step=self.global_step)


        '''Finished creating graph: start session and init vars'''
        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())


    def run_train(self, input_syllabs, target_syllabs):
        '''Call this from outside the class to run a train step'''
        cost, lr, _ = self.sess.run([self.cost, self.learning_rate, self.train_step],
                                   feed_dict={
                                       self.inputs: input_syllabs,
                                       self.targets: target_syllabs
                                   })
        return cost, lr
