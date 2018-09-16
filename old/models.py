import tensorflow as tf

from layers import EmbeddingLayer, TiedIOEmbedding, LSTM, HRRWordEmbedding, HRRChunkLayer
from datasets import PAD_ID
from utils import gather_last_d

class BaseModel(object):

    def __init__(self, **kwargs):
        self._get_params(**kwargs)
        self._build_inputs()
        self._build_layers()
        self._build_forward()
        self._build_backward()
        self._build_metrics()
        self._build_summaries()    

    def _build_inputs(self):
        raise NotImplemented()

    def _build_layers(self):
        raise NotImplemented()
        
    def _build_forward(self):
        raise NotImplemented()

    def _build_backward(self):
        raise NotImplemented()
    
    def _build_metrics(self):
        raise NotImplemented()

    def _build_summaries(self):
        for name, (metric, weight) in self.metrics.items():
            tf.summary.scalar(name, metric)
        self.summary_op = tf.summary.merge_all()

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

    def _build_layers(self):
        emb_cls = TiedIOEmbedding if self.tied_io else EmbeddingLayer
        self.emb = emb_cls(self.vocab_size, self.cell_dim)
        self.rnn = LSTM(self.num_layers, self.cell_dim, keep_prob=self.keep_prob)

    def _build_inputs(self):
        input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids') # size: bs x sl
        target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
        self.inputs = {'input_ids': input_ids, 'target_ids': target_ids}
        
    def _build_forward(self):
        input_ids = self.inputs['input_ids']
        target_ids = self.inputs['target_ids']
        mask = tf.to_float(tf.not_equal(target_ids, PAD_ID))
        
        with tf.name_scope('forward'):
            e = self.emb(input_ids)
            e = tf.nn.dropout(e, keep_prob=self.keep_prob)
            output, states = self.rnn(e)
            logits = self.emb(output, predict=True)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_ids) * mask
            
            self.num_words = tf.reduce_sum(mask)
            self.loss = tf.reduce_sum(losses) / self.num_words
    
    def _build_metrics(self):
        self.metrics = {'loss': (self.loss, self.num_words)}

    def _build_backward(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = tf.gradients(self.loss, tf.trainable_variables())
        grads, norm = tf.clip_by_global_norm(grads, 5.0) # gradient clipping
        grads_and_vars = zip(grads, tf.trainable_variables())
        self.train_op = self.optimizer.apply_gradients(grads_and_vars)

class HRRWordLM(BaseLM):

    def _get_params(self, **kwargs):
        super(HRRWordLM, self)._get_params(**kwargs)
        assert self.tied_io

        self.num_roles = kwargs['num_roles']
        self.num_fillers = kwargs['num_fillers']

    def _build_layers(self):
        self.emb = HRRWordEmbedding(self.vocab_size, self.cell_dim, self.num_roles, self.num_fillers)
        self.rnn = LSTM(self.num_layers, self.cell_dim, keep_prob=self.keep_prob)

class HRRChunkLM(HRRWordLM):
    
    def _build_layers(self):
        self.emb = HRRWordEmbedding(self.vocab_size, self.cell_dim, self.num_roles, self.num_fillers)
        self.rnn = LSTM(self.num_layers, self.cell_dim, keep_prob=self.keep_prob, proj_dim=self.cell_dim * 2)
        self.chunk_layer = HRRChunkLayer(self.cell_dim, self.num_roles)
    
    def _build_inputs(self):
        super(HRRChunkLM, self)._build_inputs()
        chunk_ids = tf.placeholder(tf.int32, shape=[None, None], name='chunk_ids')
        self.inputs['chunk_ids'] = chunk_ids

    def _build_forward(self):
        input_ids = self.inputs['input_ids']
        target_ids = self.inputs['target_ids']
        word_mask = tf.to_float(tf.not_equal(target_ids, PAD_ID))
        
        chunk_ids = self.inputs['chunk_ids']
        
        with tf.name_scope('forward'):
            e = self.emb(input_ids)
            e = tf.nn.dropout(e, keep_prob=self.keep_prob)
            output, states = self.rnn(e)
            word_output, chunk_output = tf.split(output, num_or_size_splits=2, axis=-1)
            # compute logits for words
            word_logits = self.emb(word_output, predict=True)
            word_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=word_logits, labels=target_ids) * word_mask
            # compute logits for chunks
            chunk_logits, chunk_weight = self.chunk_layer(chunk_ids, chunk_output, e)
            chunk_log_probs = tf.nn.log_softmax(chunk_logits + (1.0 - tf.reshape(chunk_weight, [-1])) * (-999.9), axis=-1)
            
            chunk_target = tf.range(tf.size(chunk_weight))
            chunk_losses = -gather_last_d(chunk_log_probs, chunk_target) * chunk_weight # NOTE: minus sign!
            
            self.num_words = tf.reduce_sum(word_mask)
            self.num_chunks = tf.reduce_sum(chunk_weight)
            self.word_loss = tf.reduce_sum(word_losses) / self.num_words
            self.chunk_loss =  tf.reduce_sum(chunk_losses) / self.num_chunks
            self.loss = self.word_loss + self.chunk_loss
    
    def _build_metrics(self):
        self.metrics = {'loss': (self.loss, tf.constant(1.0)), 
                        'word_loss': (self.word_loss, self.num_words),
                        'chunk_loss': (self.chunk_loss, self.num_chunks)}        
        