import tensorflow as tf

from utils import mm3by2, circular_conv, circular_corr, get_name

class BaseLayer(object):

    def __init__(self):
        raise NotImplemented()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        raise NotImplemented()

class Linear(object):
    
    def __init__(self, input_dim, output_dim, name=''):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        with tf.variable_scope(get_name(name, 'linear')):
            self.weight = tf.get_variable('weight', shape=[input_dim, output_dim], dtype=tf.float32)
            self.bias = tf.get_variable('bias', shape=[output_dim], dtype=tf.float32)
    
    def __call__(self, input_, tm=False):
        if tm:
            return mm3by2(input_, self.weight) + self.bias
        else:
            return tf.matmul(input_, self.weight) + self.bias

class ChunkBindingLayer(BaseLayer):
    
    def __init__(self, dim, num_roles, name=''):
        self.dim = dim
        self.num_roles = num_roles
        
        with tf.variable_scope(get_name(name, 'chunk_binding')):
            self.r_basis = tf.get_variable('r_basis', shape=[num_roles, dim], dtype=tf.float32)
            self.r_pred = Linear(dim, num_roles, name='pred')
            self.f_proj = Linear(dim, dim * 2, name='proj')
    
    def __call__(self, rnn_output, word_embedding):
        with tf.name_scope('chunk_bind'):
            bs, sl, _ = tf.unstack(tf.shape(rnn_output))
            # get role embeddings
            r_logits = tf.reshape(self.r_pred(rnn_output, tm=True), [bs, sl, self.num_roles])
            r_probs = tf.nn.softmax(r_logits)
            r = self.r_basis * tf.expand_dims(r_probs, axis=-1) # ? x nr x d
            
            # get filler embeddings
            f = tf.reshape(tf.tanh(self.f_proj(word_embedding, tm=True)), [bs, sl, self.num_roles, self.dim])
            rbf = circular_conv(r, f, name='rbf')
            # bound_e = tf.reduce_sum(rbf, axis=-2) # ? x d
            return rbf

class ChunkEncodingLayer(BaseLayer):
    
    def __init__(self, num_roles):
        self.num_roles = num_roles
    
    def __call__(self, chunk_ids, bound_word_embedding):
        with tf.name_scope('chunk_encode'):
            # chunk_ids to chunk_weight_matrix
            
            bs, sl = tf.unstack(tf.shape(chunk_ids))
            cl = sl
            batch_ind = tf.tile(tf.expand_dims(tf.range(bs), axis=1), [1, sl])
            pos_ind = tf.tile(tf.expand_dims(tf.range(sl), axis=0), [bs, 1])
            indices = tf.stack([batch_ind, chunk_ids, pos_ind], axis=-1)
            updates = tf.ones([bs, sl])
            shape = [bs, cl, sl]
            cwm = tf.scatter_nd(indices, updates, shape)
            
            # get chunk embeddings
            eis = list()
            for ri in range(self.num_roles):
                bound_e = bound_word_embedding[..., ri, :]
                ei = tf.matmul(cwm, bound_e)
                eis.append(ei)
            chunk_embedding = tf.reduce_sum(tf.stack(eis, axis=0), axis=0) # bs x cl x d
            chunk_weight = tf.to_float(tf.reduce_max(cwm, axis=-1) > 0) # NOTE the first position has weight 1.0, but this should be ignored later (after truncation)
            return chunk_embedding, chunk_weight, eis
    
class ChunkPredictionLayer(BaseLayer):
    
    def __init__(self, dim, mode='trigram', name=''):
        self.dim = dim
        self.mode = mode
        assert self.mode in ['bigram', 'trigram']
        
        with tf.variable_scope(get_name(name, 'chunk_pred')):
            if self.mode == 'bigram':
                self.pred = tf.get_variable('pred', shape=[dim, dim], dtype=tf.float32)
            else:
                self.pred = tf.get_variable('pred', shape=[dim * 2, dim], dtype=tf.float32)
        
    def __call__(self, input_):
        with tf.name_scope('chunk_pred'):
            if self.mode == 'bigram':
                res = mm3by2(e, self.pred)
            else:
                input_prev = input_[:, 1:, :]
                input_prev = tf.concat([tf.zeros_like(input_[:, :1]), input_prev], axis=1)
                cat = tf.concat([input_prev, input_], axis=-1)
                res = mm3by2(cat, self.pred)
            bs, cl, _ = tf.unstack(tf.shape(input_))
            return tf.reshape(res, [bs, cl, self.dim])

class HRRChunkLayer(BaseLayer):
    
    def __init__(self, dim, num_roles, name=''):
        self.dim = dim
        self.num_roles = num_roles
        
        with tf.variable_scope(get_name(name, 'HRR_chunk')):
            self.binder = ChunkBindingLayer(dim, num_roles)
            self.encoder = ChunkEncodingLayer(num_roles)
            self.pred = ChunkPredictionLayer(dim)
    
    def __call__(self, chunk_ids, rnn_output, word_embedding):
        with tf.name_scope('chunk_HRR'):
            bound_word_embedding = self.binder(rnn_output, word_embedding)
            chunk_embedding, chunk_weight, chunk_decomp = self.encoder(chunk_ids, bound_word_embedding)
            next_chunk_pred = self.pred(chunk_embedding)
            
            with tf.name_scope('truncate'):
                target_chunk_embedding = chunk_embedding[:, 2:]
                next_chunk_pred = next_chunk_pred[:, 1:-1]
            
            with tf.name_scope('logits'):
                logits = list()
                next_chunk_decomp = circular_corr(self.binder.r_basis, tf.expand_dims(next_chunk_pred, axis=-2))
                for ri in range(self.num_roles):
                    f_pred = next_chunk_decomp[..., ri, :] # bs x cl x d
                    f_target = chunk_decomp[ri][:, 2:] # bs x cl x d
                    f_target_flat = tf.reshape(f_target, [-1, self.dim]) # bs*cl x d
                    logit = tf.reduce_sum(tf.expand_dims(f_pred, axis=-2) * f_target_flat, axis=-1)
                    logits.append(logit)
                logits = tf.reduce_sum(tf.stack(logits, axis=0), axis=0) # bs x cl x bs*cl
                chunk_weight = chunk_weight[:, 2:]
                
        return logits, chunk_weight

class EmbeddingLayer(BaseLayer):

    def __init__(self, vocab_size, dim, name=''):
        self.vocab_size = vocab_size
        self.dim = dim
        
        with tf.variable_scope(get_name(name, 'emb')):
            self.weight = tf.get_variable("weight", shape=[self.vocab_size, self.dim], dtype=tf.float32)

    def forward(self, input_):
        return tf.nn.embedding_lookup(self.weight, input_)

class TiedIOEmbedding(EmbeddingLayer):

    def __init__(self, vocab_size, dim, use_bias=True, name=''):#False):
        name = get_name(name, 'tied_emb')
        super(TiedIOEmbedding, self).__init__(vocab_size, dim, name=name)
        self.use_bias = use_bias
        if self.use_bias:
            with tf.variable_scope(name):
                self.bias = tf.get_variable('bias', shape=[self.vocab_size], dtype=tf.float32)

    def __call__(self, input_, predict=False):
        call_func = self._predict if predict else self.forward
        return call_func(input_)

    def _predict(self, input_):
        with tf.name_scope('predict'):
            bs, sl, _ = tf.unstack(tf.shape(input_))
            mm = mm3by2(input_, self.weight, transpose_b=True)
            if self.use_bias:
                mm += self.bias
            return tf.reshape(mm, [bs, sl, self.vocab_size])

class HRRWordEmbedding(TiedIOEmbedding):

    def __init__(self, vocab_size, dim, num_roles, num_fillers, use_bias=True, name=''):
        name = get_name(name, 'HRR_word')
        super(HRRWordEmbedding, self).__init__(vocab_size, dim, use_bias=use_bias, name=name)
        self.num_roles = num_roles
        self.num_fillers = num_fillers
        
        with tf.variable_scope(name):
            self.r_basis = tf.get_variable('r_basis', shape=[num_roles, dim], dtype=tf.float32)
            self.f_basis = tf.get_variable('f_basis', shape=[num_roles, num_fillers, dim], dtype=tf.float32)
            self.s = tf.get_variable('s', shape=[vocab_size, num_roles, num_fillers], dtype=tf.float32)
            self.f = tf.reduce_sum(tf.expand_dims(self.s, axis=-1) * self.f_basis, axis=2, name='f') # size: vs x nr x d
            self.rbf = circular_conv(self.r_basis, self.f, name='rbf') # r-bind-f. size: vs x nr x d
            # replace the old weight
            self.weight = tf.reduce_sum(self.rbf, axis=1, name='weight') # size: vs x d

    def _predict(self, input_):
        with tf.name_scope('predict'):
            bs, sl, _ = tf.unstack(tf.shape(input_))
            f_noisy = circular_corr(self.r_basis, tf.expand_dims(input_, axis=-2)) # size: bs x sl x nr x d
            mm = 0.0
            for ri in range(self.num_roles):
                x = f_noisy[:, :, ri]
                w = self.f[:, ri]
                mm += mm3by2(x, w, transpose_b=True)
            if self.use_bias:
                mm += self.bias
            return tf.reshape(mm, [bs, sl, self.vocab_size])

class LSTM(BaseLayer):

    def __init__(self, num_layers, dim, keep_prob=1.0, proj_dim=None, name=''):
        self.num_layers = num_layers
        self.dim = dim
        self.proj_dim = proj_dim

        with tf.variable_scope(get_name(name, 'lstm')):
            cells = [tf.nn.rnn_cell.LSTMCell(self.dim, num_proj=proj_dim) for _ in range(self.num_layers)]
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in cells]
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    def forward(self, input_):
        return tf.nn.dynamic_rnn(cell=self.cell,
                                 inputs=input_,
                                 dtype=tf.float32)
