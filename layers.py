import tensorflow as tf

from utils import mm3by2

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

    def __init__(self, vocab_size, dim, use_bias=False):
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

class LSTM(BaseLayer):

    def __init__(self, num_layers, dim):
        self.num_layers = num_layers
        self.dim = dim

        with tf.variable_scope('lstm_cell'):
            cells = [tf.nn.rnn_cell.LSTMCell(self.dim) for _ in range(self.num_layers)]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    def forward(self, input_):
        return tf.nn.dynamic_rnn(cell=self.cell,
                                 inputs=input_,
                                 dtype=tf.float32)
