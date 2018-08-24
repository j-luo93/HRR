import tensorflow as tf

from utils import mm3by2, circular_conv, circular_corr

class BaseLayer(object):

    def __init__(self):
        raise NotImplemented()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        raise NotImplemented()

class EmbeddingLayer(BaseLayer):

    def __init__(self, vocab_size, dim):
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = tf.get_variable("weight", shape=[self.vocab_size, self.dim], dtype=tf.float32)

    def forward(self, input_):
        return tf.nn.embedding_lookup(self.weight, input_)

class TiedIOEmbedding(EmbeddingLayer):

    def __init__(self, vocab_size, dim, use_bias=True):#False):
        super(TiedIOEmbedding, self).__init__(vocab_size, dim)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = tf.get_variable('bias', shape=[self.vocab_size], dtype=tf.float32)

    def __call__(self, input_, predict=False):
        call_func = self._predict if predict else self.forward
        return call_func(input_)

    def _predict(self, input_):
        with tf.name_scope('predict'):
            mm = mm3by2(input_, self.weight, transpose_b=True)
            if self.use_bias:
                mm += self.bias
            return mm

class HRRWordEmbedding(TiedIOEmbedding):

    def __init__(self, vocab_size, dim, num_roles, num_fillers, use_bias=True):
        super(HRRWordEmbedding, self).__init__(vocab_size, dim, use_bias=use_bias)
        self.num_roles = num_roles
        self.num_fillers = num_fillers
        self.r_basis = tf.get_variable('r_basis', shape=[num_roles, dim], dtype=tf.float32)
        self.f_basis = tf.get_variable('f_basis', shape=[num_roles, num_fillers, dim], dtype=tf.float32)
        self.s = tf.get_variable('s', shape=[vocab_size, num_roles, num_fillers], dtype=tf.float32)
        self.f = tf.reduce_sum(tf.expand_dims(self.s, axis=-1) * self.f_basis, axis=2) # size: vs x nr x d
        self.rbf = circular_conv(self.r_basis, self.f) # r-bind-f. size: vs x nr x d
        # replace the old weight
        self.weight = tf.reduce_sum(self.rbf, axis=1) # size: vs x d

    def _predict(self, input_):
        with tf.name_scope('predict'):
            f_noisy = circular_corr(self.r_basis, tf.expand_dims(input_, axis=-2)) # size: bs x sl x nr x d
            mm = 0.0
            for ri in range(self.num_roles):
                x = f_noisy[:, :, ri]
                w = self.f[:, ri]
                mm += mm3by2(x, w, transpose_b=True)
            if self.use_bias:
                mm += self.bias
            return mm

class LSTM(BaseLayer):

    def __init__(self, num_layers, dim, keep_prob=1.0):
        self.num_layers = num_layers
        self.dim = dim

        with tf.variable_scope('lstm_cell'):
            cells = [tf.nn.rnn_cell.LSTMCell(self.dim) for _ in range(self.num_layers)]
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in cells]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    def forward(self, input_):
        return tf.nn.dynamic_rnn(cell=self.cell,
                                 inputs=input_,
                                 dtype=tf.float32)
