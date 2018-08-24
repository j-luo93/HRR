import tensorflow as tf

from layers import EmbeddingLayer, TiedIOEmbedding, LSTM

class BaseModel(object):

    def __init__(self, **kwargs):
        self._get_params(**kwargs)
        self._build_forward()
        self._build_backward()
        self.summary_op = tf.summary.merge_all()

    def _build_forward(self):
        raise NotImplemented()

    def _build_backward(self):
        raise NotImplemented()

    def get_inputs(self):
        assert isinstance(self._inputs, dict)
        return self._inputs

class BaseLM(BaseModel):

    def _get_params(self, **kwargs):
        self.num_layers = kwargs['num_layers']
        self.cell_dim = kwargs['cell_dim']
        self.vocab_size = kwargs['vocab_size']
        self.keep_prob = kwargs['keep_prob']
        self.tied_io = kwargs['tied_io']
        self.sample_size = kwargs['sample_size']
        self.lr = kwargs['learning_rate']
        assert self.tied_io
        assert self.sample_size == 0

    def _build_forward(self):
        self._inputs = {}
        input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids') # size: bs x sl
        target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self._inputs['input_ids'] = input_ids
        self._inputs['target_ids'] = target_ids

        bs, sl = input_ids.get_shape()

        emb_cls = TiedIOEmbedding if self.tied_io else EmbeddingLayer
        self.emb = TiedIOEmbedding(self.vocab_size, self.cell_dim)
        self.rnn = LSTM(self.num_layers, self.cell_dim, keep_prob=self.keep_prob)

        with tf.name_scope('forward'):
            e = self.emb(input_ids)
            e = tf.nn.dropout(e, keep_prob=self.keep_prob)
            output, states = self.rnn(e)
            logits = self.emb(output, predict=True)
            log_probs = tf.nn.log_softmax(logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(target_ids, [-1]))
        self.num_words = tf.reduce_sum(tf.to_float(target_ids > 0))
        self.loss = tf.reduce_sum(losses) / self.num_words
        tf.summary.scalar('loss', self.loss)

    def _build_backward(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = tf.gradients(self.loss, tf.trainable_variables())
        grads, norm = tf.clip_by_global_norm(grads, 5.0) # gradient clipping
        grads_and_vars = zip(grads, tf.trainable_variables())
        self.train_op = self.optimizer.apply_gradients(grads_and_vars)
