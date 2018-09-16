"""Common layers for language models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import tensorflow as tf

from google3.net.proto2.python.public import text_format
from google3.speech.languagemodel.sinbad import sinbad_pb2
from google3.speech.languagemodel.sinbad.contrib.tensorflow_slim.ops import sinbad_ops
from google3.third_party.tensorflow.python.framework import function
from google3.third_party.tensorflow.python.ops import functional_ops
from google3.third_party.tensorflow.python.ops import inplace_ops

from google3.third_party.tensorflow_lingvo.core import base_layer
from google3.third_party.tensorflow_lingvo.core import layers_with_attention
from google3.third_party.tensorflow_lingvo.core import recurrent
from google3.third_party.tensorflow_lingvo.core import rnn_cell
from google3.third_party.tensorflow_lingvo.tasks.lm import layers as lingvo_lm_layers
from google3.learning.brain.research.babelfish import burger_layers
from google3.learning.brain.research.babelfish import layers
from google3.learning.brain.research.babelfish import py_utils
from google3.learning.brain.research.babelfish import rnn_layers

from google3.learning.brain.research.babelfish import variational_layers
from google3.learning.brain.research.babelfish.ops import py_x_ops

# pyformat: disable
# pylint: disable=invalid-name
NullLm = lingvo_lm_layers.NullLm
MoeLm = lingvo_lm_layers.MoeLm
TransformerLmNoEmbedding = lingvo_lm_layers.TransformerLmNoEmbedding
TransformerLm = lingvo_lm_layers.TransformerLm
# pylint: enable=invalid-name
# pyformat: enable

def _RnnOutputSize(rnns):
  cell = rnns.cell_tpl[-1]
  return cell.num_output_nodes


class RnnLmNoEmbedding(lingvo_lm_layers.BaseLanguageModel):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLmNoEmbedding, cls).Params()
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    p.Define(
        'direct_features_dim', 0,
        'If > 0, then the number of dimensions of direct features '
        'that bypass the RNN and are provided directly to the softmax '
        'input.')
    p.Define('num_roles', 0, 'Number of roles')
    p.Define('decode_dropout_keep_prob', 1.0, 'Dropout after decoding. Only used for HRR')
    p.Define('num_sent_roles', 0, 'Number of top/sentence level roles')
    p.Define('global_decode', False, 'Flag to use global decode')
    # TODO(jmluo) merge these two
    p.Define('chunk_loss', False, 'Flag to use chunk loss')
    p.Define('gold_chunks', False, 'Whether gold chunks are provided')
    p.Define('cc_size', 0, 'CC size')
    p.Define('pred_mode', 'trigram', 'prediction mode')
    p.Define('chunk_input_type', 'emb', 'chunk input type')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLmNoEmbedding, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]

    cell_output_size = _RnnOutputSize(p.rnns)
    output_layer_size = cell_output_size + p.direct_features_dim

    if p.global_decode:
      if p.chunk_loss:
        output_layer_size //= 3
      else:
        output_layer_size //= 2

    actual_output_size = output_layer_size * max(1, p.num_roles)
    if actual_output_size != p.softmax.input_dim:
      raise ValueError(
          'Output layer size %d does not match softmax input size %d! '
          'cell_output_size: %d direct_features_dim: %d ' %
          (actual_output_size, p.softmax.input_dim, cell_output_size,
           p.direct_features_dim))
    if p.softmax.num_classes != p.vocab_size:
      raise ValueError(
          'softmax num of classess %d does not match vocabulary size %d!' %
          (p.softmax.num_classes, p.vocab_size))

    assert p.chunk_input_type in ['emb', 'sent_act']

    with tf.variable_scope(p.name):
      self.CreateChild('rnns', p.rnns)
      self.CreateChild('softmax', p.softmax)
      # if p.num_roles == 0:
      #   self.CreateChild('softmax', p.softmax)
      # else:
      #   # create many softmax layers
      #   for i in xrange(p.num_roles):
      #     p.softmax.name = 'softmax_%d' %i
      #     self.CreateChild('softmax_%d' %i, p.softmax)

      # get the lower level RNN
      if p.num_sent_roles > 0:
        sp = layers.SimpleFullSoftmax.Params()
        sp.name = 'lower_softmax'
        sp.num_classes = p.num_sent_roles
        input_dim = p.rnns.cell_tpl[-1].num_output_nodes
        if p.global_decode:
          if p.chunk_loss:
            input_dim = input_dim // 3
          else:
            input_dim = input_dim // 2
        sp.input_dim = input_dim # Note the output is split into two parts
        self.CreateChild('lower_softmax', sp)

        if p.global_decode:
          cc_dim = p.rnns.cell_tpl[0].num_input_nodes
          if p.pred_mode == 'bigram':
            cc_inp = cc_dim
          elif p.pred_mode == 'trigram':
            cc_inp = 2 * cc_dim
          else:
            raise
          pred_pc = py_utils.WeightParams(
            shape=[cc_inp, cc_dim], # HACK
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
          self.CreateVariable('pred', pred_pc)
          # pred_h_pc = py_utils.WeightParams(
          #   shape=[cc_dim, cc_dim], # HACK
          #   init=p.params_init,
          #   dtype=p.dtype,
          #   collections=[self.__class__.__name__ + '_vars'])
          # self.CreateVariable('pred_hidden', pred_h_pc)

          if p.chunk_loss:
            pp = layers.SimpleFullSoftmax.Params()
            pp.name = 'pred_softmax'
            pp.num_classes = p.num_sent_roles
            pp.input_dim = input_dim
            self.CreateChild('pred_softmax', pp)

          elif p.gold_chunks:
            # used for constructing two orthogonal contextualized word embeddings
            A_pc = py_utils.WeightParams(
              shape=[p.rnns.cell_tpl[0].num_input_nodes, 2 * p.rnns.cell_tpl[0].num_input_nodes], # HACK
              init=p.params_init,
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars'])
            self.CreateVariable('A', A_pc)

            # canonical chunk embeddings
            C_pc = py_utils.WeightParams(
              shape=[cc_dim, p.cc_size], # HACK
              init=p.params_init,
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars'])
            self.CreateVariable('C1', C_pc)
            self.CreateVariable('C2', C_pc)


            # CC_pc = py_utils.WeightParams(
            #   shape=[cc_dim // 10, p.rnns.cell_tpl[0].num_input_nodes], # HACK
            #   init=p.params_init,
            #   dtype=p.dtype,
            #   collections=[self.__class__.__name__ + '_vars'])
            # self.CreateVariable('CC', CC_pc)

        R_init_val = tf.random_normal(shape=[p.num_sent_roles, input_dim],
              stddev=0.044,
              dtype=tf.float32)
        R_init = py_utils.WeightInit.Constant(scale=R_init_val)
        R_pc = py_utils.WeightParams(
            shape=[p.num_sent_roles, p.rnns.cell_tpl[0].num_input_nodes], # HACK
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('R', R_pc, trainable=False)

        if not p.global_decode:
          rp = p.rnns.Copy()
          rp.name = 'lower_rnns'
          rp.num_layers = 1 # lower level rnn should be fairly simple -- just predicting roles
          rp.cell_tpl = rp.cell_tpl[:1]
          rp.cell_tpl[0].num_hidden_nodes = 0

          hsp = sp.Copy()
          hsp.name = 'higher_softmax'

          self.CreateChild('lower_rnns', rp)
          self.CreateChild('higher_softmax', hsp)


  def zero_state(self, batch_size):
    return self.rnns.zero_state(batch_size)

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def Step(self,
           theta,
           inputs,
           paddings,
           state0,
           direct_features=None,
           lower_state0=None,
           *args,
           **kwargs):
    """FProp one step.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch] or [batch, dims].
      paddings: a 0/1 tensor of shape [batch].
      state0: A NestedMap containing the initial recurrent state.
      direct_features: If not None, a tensor of[batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      output: A NestedMap with fields.
        logits: [batch, vocab_size].
        last_hidden: [batch, dims].
      state1: The new recurrent state.
    """

    def ExpandTime(x):
      return tf.expand_dims(x, axis=0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 2)
      direct_features = ExpandTime(direct_features)

    res = self.FProp(
        theta=theta,
        inputs=ExpandTime(inputs),
        paddings=ExpandTime(paddings),
        state0=state0,
        lower_state0=lower_state0,
        direct_features=direct_features,
        *args,
        **kwargs)
    p = self.params
    if p.num_sent_roles > 0 and not p.global_decode:
      assert lower_state0 is not None
      xent_output, state1, lower_state1 = res
    else:
      xent_output, state1 = res

    output = py_utils.NestedMap()
    output.logits = tf.squeeze(xent_output.logits, axis=0)
    output.probs = tf.squeeze(xent_output.probs, axis=0)
    output.log_probs = tf.squeeze(xent_output.log_probs, axis=0)
    output.last_hidden = tf.squeeze(xent_output.last_hidden, axis=0)
    if 'cce' in xent_output:
      output.cce = tf.squeeze(xent_output.cce, axis=-2)
    # TODO(jmluo) HACKY
    if 'gating_probs' in xent_output:
      output.gating_probs = tf.squeeze(xent_output.gating_probs, axis=0)

    if p.num_sent_roles > 0 and not p.global_decode:
      return output, state1, lower_state1
    else:
      return output, state1

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            lower_state0=None,
            labels=None,
            direct_features=None,
            emb_weights=None,
            chunk_ids=None,
            step_inference=False,
            ids=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input activation. A tensor of shape [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.
      direct_features: If not None, a tensor of[time, batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    paddings = py_utils.HasShape(paddings, [seqlen, batch])
    assert state0 is not None
    p = self.params

    # Storage for intermediate results
    inter_res = py_utils.NestedMap(emb_word=inputs)

    # sentence level role computation here
    if p.num_sent_roles > 0 and not p.global_decode:
      assert lower_state0 is not None
      with tf.name_scope('sent_role'):
        lower_act, lower_state1 = self.lower_rnns.FProp(theta.lower_rnns, inputs,
                                       tf.expand_dims(paddings, 2), lower_state0)


        logits = self.lower_softmax.Logits(theta=theta.lower_softmax,
                                           inputs=tf.reshape(lower_act, [seqlen * batch, -1]))
        sent_role_probs = tf.nn.softmax(logits) # sl*bs x nr
        sent_roles = py_utils.Matmul(sent_role_probs, theta.R) # sl*bs x d
        sent_roles = tf.reshape(sent_roles, [seqlen, batch, -1])
        inputs = HRREmbeddingLayer.static_circular_conv(sent_roles, inputs)

    activation, state1 = self.rnns.FProp(theta.rnns, inputs,
                                         tf.expand_dims(paddings, 2), state0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 3)
      activation = tf.concat([activation, direct_features], axis=2)

    # retrieve word level representations from the sentence level ones.
    if p.num_sent_roles > 0:
      if p.global_decode:
        with tf.name_scope('predict_sent_role'):
          if p.chunk_loss:
            sent_act, pred_act, activation = tf.split(activation, 3, axis=-1)
            pred_logits = self.pred_softmax.Logits(theta=theta.pred_softmax, inputs=tf.reshape(pred_act, [seqlen * batch, -1]))
            pred_sent_role_probs = tf.nn.softmax(pred_logits)
            pred_sent_roles = py_utils.Matmul(pred_sent_role_probs, theta.R) # sl*bs x d
            predicted_chunk = tf.matmul(tf.reshape(inputs, [seqlen * batch, -1]), theta.pred)
            predicted_chunk = tf.reshape(predicted_chunk, [seqlen, batch, -1])
          else:
            sent_act, activation = tf.split(activation, 2, axis=-1)
            inter_res.h_word = activation
            inter_res.h_sent = sent_act
          lower_logits = self.lower_softmax.Logits(theta=theta.lower_softmax, inputs=tf.reshape(sent_act, [seqlen * batch, -1]))
          lower_sent_role_probs = tf.nn.softmax(lower_logits)
          inter_res.logits_sent = lower_logits
          inter_res.role_probs = lower_sent_role_probs

          # sanity check -- one role only
          # lower_sent_role_probs = tf.stack([tf.ones([seqlen * batch]), tf.zeros([seqlen * batch])], axis=-1)

          # lower_sent_roles = py_utils.Matmul(lower_sent_role_probs, theta.R) # sl*bs x d
      else:
        with tf.name_scope('decode_sent_role'):
          higher_logits = self.higher_softmax.Logits(theta=theta.higher_softmax,
                                               inputs=tf.reshape(activation, [seqlen * batch, -1]))
          higher_sent_role_probs = tf.nn.softmax(higher_logits) # sl*bs x nr
          higher_sent_roles = py_utils.Matmul(higher_sent_role_probs, theta.R) # sl*bs x d
          higher_sent_roles = tf.reshape(higher_sent_roles, [seqlen, batch, -1])
          activation = HRREmbeddingLayer.static_decode(activation, higher_sent_roles)

    def forward(softmax_name, act, h=None): # TODO(jmluo) may wanna rename activation
      softmax_layer = getattr(self, softmax_name)
      softmax_theta = getattr(theta, softmax_name)
      if labels is None:
        # We can only compute the logits here.
        logits = softmax_layer.Logits(
            theta=softmax_theta,
            inputs=tf.reshape(act, [seqlen * batch, -1]),
            activation=h,
            return_gating=p.softmax.gating)
        if p.softmax.gating:
          logits, gating_probs = logits

        xent_output = py_utils.NestedMap(
            logits=tf.reshape(logits, [seqlen, batch, -1]))
        xent_output.probs = tf.nn.softmax(xent_output.logits)
        xent_output.log_probs = tf.nn.log_softmax(xent_output.logits)
        if p.softmax.gating:
          xent_output.gating_probs = tf.reshape(gating_probs, [seqlen, batch, -1])
      elif 'class_ids' in labels:
        print(softmax_layer)
        xent_output = softmax_layer.FProp(
            theta=softmax_theta,
            inputs=act,
            class_weights=labels.class_weights,
            class_ids=labels.class_ids,
            activation=h)
      else:
        assert 'class_probabilities' in labels
        xent_output = softmax_layer.FProp(
            theta=softmax_theta,
            inputs=act,
            class_weights=labels.class_weights,
            class_probabilities=labels.class_probabilities,
            activation=h)
      xent_output.last_hidden = activation
      return xent_output

    p = self.params
    if p.num_roles == 0:
      return forward('softmax', activation), state1
    else:
      assert emb_weights is not None
      outputs = list()
      keys = set()
      # old code without concatenating all softmax weights together
      # for i in xrange(p.num_roles):
      #   # TODO(jmluo) This is super hacky.
      #   act = self.emb.decode(activation, role_ind=i)
      #   xent_output = forward('softmax_%d' %i, act)
      #   outputs.append(xent_output)
      #   if len(keys) == 0:
      #     keys = set(xent_output.keys())
      #   else:
      #     assert keys == set(xent_output.keys())

      preceding_shape = tf.shape(activation)[:-1]
      f_noisy = self.emb.decode(tf.expand_dims(activation, axis=-2), emb_weights.r) # This is actually a bit hacky -- you don't know you have emb attribute
      if p.decode_dropout_keep_prob < 1.0 and not p.is_eval:
        f_noisy = tf.nn.dropout(f_noisy, p.decode_dropout_keep_prob)
      cat = tf.reshape(f_noisy, tf.concat([preceding_shape, [p.softmax.input_dim]], axis=0))
      out = forward('softmax', cat, h=activation)

      last_dim = tf.shape(sent_act)[-1]
      if p.chunk_input_type == 'sent_act':
        w = tf.reshape(tf.matmul(tf.reshape(sent_act, [-1, last_dim]), theta.A), [-1, 2, last_dim])
      else:
        w = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, last_dim]), theta.A), [-1, 2, last_dim])
        # w = tf.tile(tf.expand_dims(tf.reshape(inputs, [-1, last_dim]), axis=-2), [1, 2, 1])
      rw = HRREmbeddingLayer.static_circular_conv(theta.R, w)

      inter_res.w = w
      inter_res.rw = rw

      clean_w = tf.expand_dims(lower_sent_role_probs, axis=-1) * w # size: sl*bs x 2 x d
      clean_w = tf.transpose(tf.reshape(clean_w, [seqlen, batch, 2, last_dim]), perm=[1, 2, 0, 3]) # size: bs x 2 x sl x d
      out.cce = clean_w

      inter_res.w_clean = clean_w

      if p.num_sent_roles > 0:
        if p.global_decode:
          out.lower_roles = lower_sent_role_probs
          out.emb = inputs
          if p.chunk_loss:
            out.pred_roles = pred_sent_role_probs
            out.predicted_chunk = predicted_chunk
          elif p.gold_chunks and not step_inference: # skip chunk loss in step inference mode
            with tf.name_scope('chunk_prediction'):
              # chunk_ids, last_word_marks = chunk_ids

              inter_res.chunk_ids = chunk_ids

              bs_indices = tf.tile(tf.expand_dims(tf.range(batch), axis=0), [seqlen, 1])
              sl_indices = tf.tile(tf.expand_dims(tf.range(seqlen), axis=1), [1, batch])
              indices = tf.stack([bs_indices, chunk_ids, sl_indices], axis=-1) # size: sl x bs x 3
              sm_shape = [batch, seqlen, seqlen] # size: bs x cl x sl
              ones = tf.ones_like(chunk_ids)
              sm = tf.to_float(tf.scatter_nd(indices, ones, sm_shape))

              # TODO(jmluo): I don't even remember what sm stands for. Summation matrix?
              inter_res.sm = sm

#               last_word_chunk = chunk_ids * last_word_marks
#               last_word_indices = tf.stack([bs_indices, last_word_chunk, sl_indices], axis=-1)
#               sm_last_word = tf.scatter_nd(last_word_indices, sl_indices, sm_shape)
#               last_word_pos = tf.reduce_max(sm_last_word, axis=-1)
#               # tf.assert_equal(last_word_pos, tf.reduce_sum(sm_last_word, axis=-1))
#
#               id_indices = tf.stack([last_word_pos, tf.transpose(bs_indices)], axis=-1)
#               last_word_ids = tf.gather_nd(ids, id_indices)

              bound_w = tf.reduce_sum(tf.expand_dims(lower_sent_role_probs, axis=-1) * rw, axis=-2) # size: sl*bs x d
              bound_w = tf.transpose(tf.reshape(bound_w, [seqlen, batch, last_dim]), perm=[1, 0, 2]) # size: bs x sl x d
              chunk_emb = tf.matmul(sm, bound_w, name='chunk_e') # size: bs x cl x d
              chunk_emb = tf.nn.l2_normalize(chunk_emb)
              clean_chunk_emb = tf.matmul(tf.tile(tf.expand_dims(sm, axis=1), [1, 2, 1, 1]), clean_w, name='chunk_f') # size: bs x 2 x cl x d
              clean_chunk_emb = tf.nn.l2_normalize(clean_chunk_emb)

              inter_res.bound_w = bound_w
              inter_res.ce = chunk_emb
              inter_res.cce = clean_chunk_emb

              def project_to_cc_emb(embs, Cs, mode='softmax', last_word_id=None, straight_through=True):
                assert mode in ['softmax', 'argmax', 'last_word_id']
                assert mode == 'softmax'
                e1, e2 = embs
                C1, C2 = Cs
                with tf.name_scope('proj2cc'):
                  if mode == 'last_word_id':
                    assert last_word_id is not None
                    cc1 = tf.transpose(tf.gather(C1, last_word_id, axis=1), perm=[1, 2, 0])
                    cc2 = tf.transpose(tf.gather(C2, last_word_id, axis=1), perm=[1, 2, 0])
                  else:
                    logits1 = mm3by2(e1, C1) # size: bs x cl x cc
                    logits2 = mm3by2(e2, C2)
                    logits = logits1 #+ logits2
                    global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
                    # temperature = 1.0 #tf.minimum(tf.constant(200.0), global_step) / 200 * (-0.9) + 1.0
                    temperature = tf.minimum(tf.constant(1000.0), global_step) / 1000 * (-9.) + 10.0
                    tf.summary.scalar('temperature', temperature)
                    # probs = tf.nn.softmax(logits / temperature, axis=-1) # bs x cl x cc

                    def sample_gumbel(shape, eps=1e-20):
                        """Sample from Gumbel(0, 1)"""
                        U = tf.random_uniform(shape,minval=0,maxval=1)
                        return -tf.log(-tf.log(U + eps) + eps)

                    def gumbel_softmax_sample(logits, temperature):
                      """ Draw a sample from the Gumbel-Softmax distribution"""
                      y = logits + sample_gumbel(tf.shape(logits))
                      return tf.nn.softmax( y / temperature)

                    logits = tf.nn.log_softmax(logits, axis=-1)
                    y = gumbel_softmax_sample(logits, temperature)
                    if straight_through:
                      k = tf.shape(logits)[-1]
                      y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=1,keep_dims=True)), y.dtype) + 1e-8
                      y = tf.stop_gradient(y_hard - y) + y
                    probs = y

                    if mode == 'softmax':
                      cc1 = mm3by2(probs, C1, transpose=True) # size: bs x cl x d
                      cc2 = mm3by2(probs, C2, transpose=True)
                    else:
                      nearest_index = tf.argmax(logits, axis=-1) # size: bs x cl
                      # TODO(jmluo) get rid of tf.transpose
                      cc1 = tf.transpose(tf.gather(C1, nearest_index, axis=1), perm=[1, 2, 0])
                      cc2 = tf.transpose(tf.gather(C2, nearest_index, axis=1), perm=[1, 2, 0])
                  return cc1, cc2, probs
                  # logits = mm3by2(emb, C) # size: bs x cl x cc
                  # probs = tf.nn.softmax(logits, axis=-1)
                  # if mode == 'softmax':
                  #   return mm3by2(probs, C, transpose=True) # size: bs x cl x d
                  # else:
                  #   nearest_index = tf.argmax(logits, axis=-1) # size: bs x cl
                  #   # TODO(jmluo) get rid of tf.transpose
                  #   return tf.transpose(tf.gather(C, nearest_index, axis=1), perm=[1, 2, 0])

              def mm3by2(x, y, transpose=False):
                with tf.name_scope('mm3by2'):
                  py_utils.HasRank(x, 3)
                  py_utils.HasRank(y, 2)
                  bs, sl, dx = tf.unstack(tf.shape(x))
                  dy = tf.shape(y)[0 if transpose else 1]
                  return tf.reshape(tf.matmul(tf.reshape(x, [bs * sl, dx]), y, transpose_b=transpose), [bs, sl, dy])

              def get_predictions(chunk_emb):
                if p.pred_mode == 'bigram':
                  cat = chunk_emb
                elif p.pred_mode == 'trigram':
                  # note that length dim is the second axis
                  bs, cl, d = tf.unstack(tf.shape(chunk_emb))
                  prev = tf.concat([tf.zeros([bs, 1, d]), chunk_emb[:, :-1]], axis=1)
                  cat = tf.concat([prev, chunk_emb], axis=-1)
                h_chunk = mm3by2(tf.tanh(cat), theta.pred) # size: bs x cl x d
                return h_chunk

              # pred before cleanup
              h_chunk = get_predictions(chunk_emb)
              f_chunk = HRREmbeddingLayer.static_circular_corr(theta.R, tf.expand_dims(h_chunk, axis=-2)) # size: bs x cl x 2 x d
              f_hat1, f_hat2 = tf.unstack(f_chunk[:, 1:-1], axis=-2)

              inter_res.h_chunk = h_chunk
              inter_res.f_chunk = f_chunk
              inter_res.f_hat1 = f_hat1
              inter_res.f_hat2 = f_hat2
              # # no cleanup
              # cc1, cc2 = tf.unstack(f_chunk[:, 1:-1], axis=-2)
              # cc1, cc2, _ = project_to_cc_emb(tf.unstack(f_chunk[:, 1:-1], axis=-2), [theta.C1, theta.C2])

              # # pred after cleanup
              # f_chunk = HRREmbeddingLayer.static_circular_corr(theta.R, tf.expand_dims(chunk_emb, axis=-2)) # size: bs x cl x 2 x d
              # cc1, cc2, _ = project_to_cc_emb(tf.unstack(f_chunk[:, 1:-1], axis=-2), [theta.C1, theta.C2])
              # cc1 = mm3by2(tf.tanh(cc1), theta.pred)
              # cc1 = project_to_cc_emb(f_chunk[..., 0, :], theta.C1, mode='softmax')[:, 1:-1]
              # cc2 = project_to_cc_emb(f_chunk[..., 1, :], theta.C2, mode='softmax')[:, 1:-1]
              # new
              # CR1 = tf.matmul(theta.CC, theta.C1, transpose_a=True)
              # CR2 = tf.matmul(theta.CC, theta.C2, transpose_a=True)
              # cc1, cc2 = project_to_cc_emb(tf.unstack(f_chunk[:, 1:-1], axis=-2), [CR1, CR2], last_word_id=last_word_ids[:, 1:-1])
              # cc1 = mm3by2(tf.tanh(mm3by2(tf.tanh(cc1), theta.pred)), theta.pred_hidden)
              # cc2 = mm3by2(cc2, theta.pred)
              # # CC
              # dot1 = mm3by2(cc1, theta.C1)
              # contrastive estimation
              gold1, gold2 = tf.unstack(clean_chunk_emb[:, :, 2:], axis=1)
              merged_indices = tf.reshape(tf.range(batch * (seqlen - 2)), [batch, -1])
              dot1 = mm3by2(f_hat1, tf.reshape(gold1, [batch * (seqlen - 2), -1]), transpose=True) # bs x cl x bs*cl
              dot2 = mm3by2(f_hat2, tf.reshape(gold2, [batch * (seqlen - 2), -1]), transpose=True) # bs x cl x bs*cl
              global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
              temperature = tf.minimum(tf.constant(50000.0), global_step) / 50000
              tf.summary.scalar('temperature', temperature)
              # dot2 = mm3by2(cc2, CR2)
              den_dot = dot1 + dot2 * temperature

              inter_res.gold1 = gold1
              inter_res.gold2 = gold2
              inter_res.dot1 = dot1
              inter_res.dot2 = dot2
              inter_res.dot = den_dot

              # clean_cc1 = project_to_cc_emb(clean_chunk_emb[:, 0], theta.C1)[:, 2:]
              # clean_cc2 = project_to_cc_emb(clean_chunk_emb[:, 1], theta.C2)[:, 2:]
              # # old
              # clean_cc1, clean_cc2, probs = project_to_cc_emb(tf.unstack(clean_chunk_emb[:, :, 2:], axis=1), [theta.C1, theta.C2])
              # new
              # clean_cc1, clean_cc2 = project_to_cc_emb(tf.unstack(clean_chunk_emb[:, :, 2:], axis=1), [CR1, CR2], last_word_id=last_word_ids[:, 2:])

              with tf.name_scope('chunk_loss'):
                # num_dot = tf.reduce_sum(clean_cc1 * cc1, axis=-1)# + clean_cc2 * cc2, axis=-1)
                # chunk_log_probs = num_dot - tf.reduce_logsumexp(den_dot, axis=-1) # size: bs x cl
                # chunk_log_probs = tf.reduce_sum(probs * tf.nn.log_softmax(den_dot), axis=-1)
                # chunk_weights = tf.transpose(tf.to_float(chunk_ids[2:] > 0))
                chunk_weights = tf.to_float(tf.reduce_max(sm, axis=-1) > 0)[:, 2:] # size: bs x cl
                one_hot_target = tf.one_hot(merged_indices, batch * (seqlen - 2), off_value=1e-8)
                den_dot = den_dot + tf.reshape(chunk_weights * 99.0 - 99.0, [-1])
                chunk_log_probs = tf.reduce_sum(one_hot_target * tf.nn.log_softmax(den_dot), axis=-1)
                out.chunk_log_probs = chunk_log_probs * chunk_weights
                out.num_chunks = tf.reduce_sum(chunk_weights)
                # out.cc_probs = probs * tf.expand_dims(chunk_weights, axis=-1) # size: bs x cl x cc

                inter_res.w_chunk = chunk_weights
                inter_res.target = one_hot_target
                inter_res.masked_dot = den_dot
                inter_res.clp = out.chunk_log_probs
                inter_res.num_chunks = out.num_chunks
          out.inter_res = inter_res
          return out, state1
        else:
          out.lower_roles = sent_role_probs
          out.higher_roles = higher_sent_role_probs
          return out, state1, lower_state1
      else:
        return out, state1
#       # logits and log_probs are always computed.
#       logits = sum([o.logits for o in outputs])
#       log_probs = tf.nn.log_softmax(logits)
#       probs = tf.exp(log_probs)
#       if labels is None:
#         return py_utils.NestedMap(logits=logits,
#                                   probs=probs,
#                                   log_probs=log_probs,
#                                   last_hidden=activation), state1
#       elif 'class_probabilities' in labels:
#         per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
#           labels=labels.class_probabilities, logits=logits)
#       elif 'class_ids' in labels:
#         per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#           labels=labels.class_ids, logits=logits)
#       else:
#         print(p)
#         raise
#
#       per_example_argmax = py_utils.ArgMax(logits)
#       label_weights = labels.class_weights
#       total_xent = tf.reduce_sum(per_example_xent * label_weights)
#       total_weight = tf.reduce_sum(label_weights)
#       avg_xent = total_xent / total_weight
#
#       res = py_utils.NestedMap(logits=logits,
#                                probs=probs,
#                                log_probs=log_probs,
#                                per_example_xent=per_example_xent,
#                                per_example_argmax=per_example_argmax,
#                                last_hidden=activation,
#                                avg_xent=avg_xent,
#                                total_weight=total_weight,
#                                total_xent=total_xent)
#       return res, state1

class RnnLmNoEmbeddingVRNN(lingvo_lm_layers.BaseLanguageModel):
  """Stacked RNN based language model layer + VRNN."""

  @classmethod
  def Params(cls):
    p = super(RnnLmNoEmbeddingVRNN, cls).Params()
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    # VRNN params.
    p.Define('var_layer', variational_layers.FuncVRNN.Params(),
             'Variational layer.')
    p.Define('var_mode', 'filter', 'Mode of VRNN.')
    p.Define('latent_dim',
             variational_layers.VRAE.Params().latent_dim.Copy(),
             'Dimension of latent variables.')
    p.Define('context_dim', 0, 'Dimension of the context, if available.')
    p.Define(
        'concat_z', True, 'If True, concat z with softmax input. '
        'Otherwise, add z to softmax input.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLmNoEmbeddingVRNN, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]

    py_utils.SetNameIfNone(p.var_layer, 'vrnn')

    assert p.rnns.cell_tpl[-1].num_output_nodes == p.softmax.input_dim, (
        '{} vs. {}'.format(p.rnns.cell_tpl[-1].num_output_nodes,
                           p.softmax.input_dim))
    assert p.softmax.num_classes == p.vocab_size, ('{} vs. {}'.format(
        p.softmax.num_classes, p.vocab_size))

    with tf.variable_scope(p.name):
      self.CreateChild('rnns', p.rnns)

      # Variational layer on top of RNN.
      params = p.var_layer
      params.name = 'vrnn'
      params.mode = p.var_mode
      params.input_dim = p.rnns.cell_tpl[-1].num_output_nodes
      params.latent_dim = p.latent_dim
      params.rnn_cell_dim = p.rnns.cell_tpl[-1].num_output_nodes
      # For LM task no src info needed.
      params.src_info_dim = p.context_dim
      if params.mode == 'filter':
        params.tgt_info_dim = p.rnns.cell_tpl[0].num_input_nodes
      self.CreateChild('vrnn_layer', params)

      ld = variational_layers.ParseLatentDim(p.latent_dim)
      sample_dim = ld.latent_dim
      if p.var_layer.sample_dim > 0:
        sample_dim = p.var_layer.sample_dim

      if p.concat_z:
        p.softmax.input_dim += sample_dim
      else:
        assert sample_dim == p.softmax.input_dim, (
            'Latent sample dim is different from softmax input_dim, cannot '
            'merge! (%s vs %s)' % (sample_dim, p.softmax.input_dim))
      self.CreateChild('softmax', p.softmax)

  def zero_state(self, batch_size):
    return self.rnns.zero_state(batch_size)

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def FProp(self, theta, inputs, paddings, state0, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input activation. A tensor of shape [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    paddings = py_utils.HasShape(paddings, [seqlen, batch])
    assert state0 is not None
    activation, state1 = self.rnns.FProp(theta.rnns, inputs,
                                         tf.expand_dims(paddings, 2), state0)
    paddings = tf.expand_dims(paddings, 2)
    p = self.params
    # activation: [time, batch, dim], paddings: [time, batch, 1]
    # Apply VRNN on top.
    if p.var_mode == 'predictor':
      activation, zs, kl_loss, _, _ = self.vrnn_layer.FProp(
          theta.vrnn_layer, activation, paddings)
    elif p.var_mode == 'filter':
      # Rotate inputs by one time step.
      tgt_info = tf.concat([
          tf.slice(inputs, [1, 0, 0], [-1, -1, -1]),
          tf.slice(inputs, [0, 0, 0], [1, -1, -1])
      ], 0)
      activation, zs, kl_loss, _, _ = self.vrnn_layer.FProp(
          theta.vrnn_layer, activation, [tgt_info, paddings])

    zs = zs.out
    num_samples = p.var_layer.sample_size
    if p.is_eval:
      num_samples = 1
    activation = tf.tile(activation, [1, num_samples, 1])
    if p.concat_z:
      activation_zs = tf.concat([activation, zs], 2)
    else:
      activation_zs = activation + zs
    if labels is None:
      # We can only compute the logits here.
      logits = self.softmax.Logits(
          theta=theta.softmax,
          inputs=tf.reshape(activation_zs, [seqlen * batch * num_samples, -1]))
      xent_output = py_utils.NestedMap(
          logits=tf.reshape(logits, [seqlen, batch, -1]))
    elif 'class_ids' in labels:
      # labels.class_ids: [len, batch]
      class_ids = tf.tile(labels.class_ids, [1, num_samples])
      class_weights = tf.tile(labels.class_weights, [1, num_samples])
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=activation_zs,
          class_weights=class_weights,
          class_ids=class_ids)
    else:
      assert 'class_probabilities' in labels
      class_probabilities = tf.tile(labels.class_probabilities,
                                    [1, num_samples])
      class_weights = tf.tile(labels.class_weights, [1, num_samples])
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=activation_zs,
          class_weights=class_weights,
          class_probabilities=class_probabilities)
    xent_output.last_hidden = activation
    xent_output.kl_loss = kl_loss
    return xent_output, state1


class RnnLm(RnnLmNoEmbedding):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLm, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.Define('embedding_dropout_keep_prob', 1.0, 'Embedding dropout keep prob.')
    p.Define('embedding_dropout_seed', None, 'Embedding dropout seed.')
    p.Define('tie', False, 'Tie input and output embeddings')
    p.Define('two_copies', False, 'Use two copies for embeddings (baseline)')
    p.emb.max_num_shards = 1
    return p

  # TODO(zhifengc): Consider merge Params() and CommonParams().
  @classmethod
  def CommonParams(cls,
                   vocab_size,
                   emb_dim=1024,
                   num_layers=2,
                   rnn_dims=2048,
                   rnn_hidden_dims=0,
                   residual_start=1,
                   softmax_max_alloc=None):
    """A LM model parameterized by vocab size, etc.

    Args:
      vocab_size: Vocab size.
      emb_dim: Embedding dimension.
      num_layers: The number of rnn layers.
      rnn_dims: Each RNN layer has this many output nodes.
      rnn_hidden_dims: If > 0, each RNN layer has this many hidden nodes.
      residual_start: index of the first layer with a residual connection;
        higher index layers also have residuals.
      softmax_max_alloc: If set to a positive integer the soft-max
        computation is chunked into allocations of at most
        softmax_max_alloc; when left to its default value of None no
        chunking is done.

    Returns:
      A RnnLm parameter object.
    """
    p = cls.Params()
    p.vocab_size = vocab_size

    init_scale = 1.0 / math.sqrt(rnn_dims)

    # Embedding.
    p.emb.vocab_size = vocab_size
    p.emb.embedding_dim = emb_dim
    p.emb.scale_sqrt_depth = True
    p.emb.params_init = py_utils.WeightInit.Uniform(init_scale)

    # RNNs
    p.rnns.num_layers = num_layers
    # Which layer starts to have the residual connection.
    p.rnns.skip_start = residual_start
    if num_layers > 1:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims),
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=rnn_dims,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]
    else:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]

    # Softmax
    p.softmax.input_dim = rnn_dims
    p.softmax.num_classes = vocab_size
    p.softmax.params_init = py_utils.WeightInit.Uniform(init_scale)
    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.rnns.cell_tpl[0].num_input_nodes, (
        '{} vs. {}'.format(p.emb.embedding_dim,
                           p.rnns.cell_tpl[0].num_input_nodes))

    if p.tie:
      assert p.emb.actual_shards == p.softmax.num_shards

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

    if p.two_copies:
      with tf.variable_scope(p.name + '_secondary'):
        p.emb.name += '_secondary'
        self.CreateChild('emb_sec', p.emb)

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            lower_state0=None,
            labels=None,
            direct_features=None,
            chunk_ids=None,
            step_inference=False,
            ids=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.
      direct_features: If not None, a tensor of[time, batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    assert state0

    p = self.params

    def forward(activation):
      # Dropout on embeddings is only applied in training.
      if p.embedding_dropout_keep_prob < 1.0 and not p.is_eval:
        activation = tf.nn.dropout(
            activation,
            keep_prob=p.embedding_dropout_keep_prob,
            seed=p.embedding_dropout_seed)

      return super(RnnLm, self).FProp(theta, activation, paddings, state0,
                                      labels=labels,
                                      lower_state0=lower_state0,
                                      direct_features=direct_features,
                                      emb_weights=emb_weights,
                                      chunk_ids=chunk_ids,
                                      step_inference=step_inference,
                                      ids=ids)


    # TODO(jmluo) may wanna get rid of this assertion to obtain a baseline (nr > 0 but w/o HRR)
    # also, should move this into __init__.
    if p.num_roles > 0:
      assert p.emb.cls == HRREmbeddingLayer
      assert p.tie

    if p.emb.cls == HRREmbeddingLayer:
      activation, signature, emb_weights = self.emb.EmbLookup(theta.emb, ids, role_anneal=p.softmax.role_anneal)
    else:
      activation = self.emb.EmbLookup(theta.emb, ids)
      if p.two_copies:
        act_sec = self.emb_sec.EmbLookup(theta.emb_sec, ids)
        global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
        temperature = tf.minimum(tf.constant(3000.0), global_step) / 3000
        activation = activation + temperature * act_sec
      emb_weights = None

    if p.two_copies:
      assert p.tie

    if p.tie:
      num_shards = p.emb.actual_shards

      def transpose_or_not(w):
        transpose = (p.softmax.num_sampled == 0)
        if transpose:
          return tf.transpose(w)
        else:
          return w

      if p.emb.cls == HRREmbeddingLayer:
        if p.num_roles > 0:
          # for i in xrange(p.num_roles):
          #   softmax_theta = getattr(theta, 'softmax_%d' %i)
          for shard_ind in xrange(num_shards):
            f_shard = emb_weights.f[shard_ind]
            reshaped_f_shard = tf.reshape(f_shard, [-1, p.softmax.input_dim])
            theta.softmax['weight_%d' %shard_ind] = transpose_or_not(reshaped_f_shard)
        else:
          for shard_ind in xrange(num_shards):
            theta.softmax['weight_%d' %shard_ind] = transpose_or_not(emb.e[shard_ind])
      else:
        for shard_ind in xrange(num_shards):
          main = transpose_or_not(theta.emb.wm[shard_ind])
          if p.two_copies:
            sec = transpose_or_not(theta.emb_sec.wm[shard_ind])
            theta.softmax['weight_%d' %shard_ind] = main + temperature * sec
          else:
            theta.softmax['weight_%d' %shard_ind] = main

    res = forward(activation)
    xent_output = res[0]
    if 'signature' in locals():
      xent_output.signature = signature
    return res


class RnnLmGraph(lingvo_lm_layers.BaseLanguageModel):
  """Language model with graph architecture."""

  @classmethod
  def Params(cls):
    p = super(RnnLmGraph, cls).Params()
    p.Define(
        'nodes', [], 'An N-dimensional array of BurgerLayer params '
        'objects specifying the decoder nodes (each node is a layer.)')
    p.Define(
        'topology', [], 'A list of tuples specifying the topology of the '
        'decoder graph. For example: [(1, 3), (2, 5)] means in the graph '
        'nodes[1] -> nodes[3], nodes[2] -> nodes[5] are edges.')
    p.Define(
        'auto_adjust', True, 'If False, an error will be raised if the '
        'combination of param specifications does not permit a valid '
        'stack (in this case it is the user\'s responsibility to ensure '
        'layer compatiblity). If True, params will be adjusted '
        'automatically (with best effort) to construct a legal layer '
        'stack.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    p.Define('dropout_prob', 0.0, 'Prob at which we do dropout.')
    p.Define('label_smoother', None, 'Label smoothing class.')
    p.Define(
        'cc_schedule', None, 'Clipping cap schedule, e.g. '
        'quant_utils.LinearClippingCapSchedule.Params().')
    p.Define(
        'per_word_loss', False,
        'If True, compute avg per-word loss. Otherwise, compute avg'
        'per-sequence loss.')
    p.Define(
        'random_seed', None,
        'If set, this decides the random seed to apply in various random'
        ' ops such that this decoder is deterministic. Set this'
        ' random_seed only for unittests.')
    p.Define('input_is_embedding', False, 'If True, then assume the input is '
             'an embedding sequence.')
    p.Define('embedding_dim', 0, 'If input_is_embedding=True, then specify '
             'embedding_dim.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLmGraph, self).__init__(params)
    p = self.params
    assert p.nodes, 'You didn\'t specify the nodes for your graph LM'
    assert p.topology, ('You didn\'t specify the topology for your graph LM')

    self._in_nodes, self._out_nodes, self._visit_plan = (
        burger_layers.GraphUtils.ParseTopology(p))

    with tf.variable_scope(p.name):
      self._node_out_dims = {}
      self._need_father_proj = set()
      emb_dim = 0
      first_in_node = True
      graph_params = []
      for (i, nf) in enumerate(self._visit_plan):
        nid, fids = nf
        node_param = p.nodes[nid].Copy()

        if node_param.ingredient.cls in (
            burger_layers.FRNNBurgerDecLayer,
            burger_layers.GatedConvBurgerDecLayer,
            burger_layers.TransformerBurgerDecLayer,
            burger_layers.FeedForwardBurgerDecLayer):
          tf.logging.error('We do not use BurgerDecLayer %s for this model!',
                           node_param.ingredient.cls)

        # Make sure self-attention is masked
        if node_param.ingredient.cls == burger_layers.SelfAttentionBurgerLayer:
          node_param.ingredient.is_masked = True
        if node_param.ingredient.cls == burger_layers.TransformerBurgerLayer:
          node_param.ingredient.mask_self_atten = True

        if nid in self._in_nodes:
          if not p.input_is_embedding:
            # To simplify the architecture, we assume input dims from each
            # channel are the same.
            node_dim = node_param.ingredient.embedding_dim
            if first_in_node:
              emb_dim = node_dim
              first_in_node = False
            assert emb_dim == node_dim, (
                'Let\'s use consistent dimensions for all embedding layers '
                '(%d vs %d)' % (emb_dim, node_dim))
            # For embedding layer, this is just a placeholder.
            if not self._node_out_dims.get(-1):
              self._node_out_dims[-1] = node_param.ingredient.embedding_dim
          else:
            # Identity layer.
            if not self._node_out_dims.get(-1):
              self._node_out_dims[-1] = p.embedding_dim

        # Check if fathers have consistent dimensions. If not, project them to
        # the same dim so that they can be merged (if auto_adjust is enabled).
        need_adjust = False
        in_dim = self._node_out_dims[fids[0]]
        father_dims = []
        for f in fids:
          dim = self._node_out_dims[f]
          father_dims.append(dim)
          if in_dim != dim:
            need_adjust = True

        if need_adjust:
          if not p.auto_adjust:
            raise ValueError(
                'Node %s receives multiple inputs from %s, but they don\'t '
                'all have the same dimensions (%s)!' % (nid, fids, father_dims))
          else:
            out_dim = min(father_dims)
            tf.logging.warning(
                'Node %s receives multiple inputs from %s, but they don\'t '
                'all have the same dimension (%s). Projecting them to the '
                'same dimension %s!', nid, fids, father_dims, out_dim)
            self._need_father_proj.add(nid)
            for d in fids:
              if self._node_out_dims[d] != out_dim:
                proj_p = p.proj_tpl.Copy().Set(
                    name='proj_%s_for_%s' % (d, nid),
                    batch_norm=False,
                    weight_norm=True,  # By default we turn on weight_norm.
                    input_dim=self._node_out_dims[d],
                    output_dim=out_dim)
                proj_p.has_bias = True
                proj_p.activation = 'NONE'
                self.CreateChild('proj_%s_for_%s' % (d, nid), proj_p)
              else:
                self.CreateChild(
                    'proj_%s_for_%s' % (d, nid),
                    layers.IdentityLayer.Params().Set(name='proj_%s_for_%s' %
                                                      (d, nid)))
            in_dim = out_dim

        # Check params first to make sure the specs are legal.
        out_dim = node_param.ingredient.pre_check(node_param.ingredient, in_dim,
                                                  i, p.auto_adjust)
        self._node_out_dims[nid] = out_dim
        node_param.dropout_prob = p.dropout_prob
        node_param.random_seed = p.random_seed
        self._node_out_dims[nid] = out_dim
        graph_params.append(node_param)
      self.CreateChildren('graph_layers', graph_params)

      # Merge outputs.
      for o in self._out_nodes:
        # Project all out nodes to p.softmax.input_dim if needed.
        out_dim = self._node_out_dims[o]
        if out_dim != p.softmax.input_dim:
          proj_p = p.proj_tpl.Copy().Set(
              name='proj_%s_for_out' % o,
              batch_norm=False,
              weight_norm=True,  # By default we turn on weight_norm.
              input_dim=out_dim,
              output_dim=p.softmax.input_dim)
          proj_p.has_bias = True
          proj_p.activation = 'NONE'
          self.CreateChild('proj_%s_for_out' % o, proj_p)

      # Softmax
      self.CreateChild('softmax', p.softmax)

      merger_p = layers_with_attention.MergerLayer.Params()
      merger_p.merger_op = 'sum'
      merger_p.name = 'merger'
      self.CreateChild('merger_layer', merger_p)

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def zero_state(self, batch_size):
    # Just a placeholder.
    return py_utils.NestedMap(
        m=tf.zeros([batch_size, 0], dtype=self.params.dtype))

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input ids.
        If not p.input_is_embedding: An int32 tensor of shape [time, batch],
        otherwise a tensor of shape [time, batch, dim].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: Not used for graph LM.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.

    Returns:
      If labels is not None, returns xent_output, where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value. Otherwise, xent_output only contains the softmax logits.
    """
    p = self.params
    if p.input_is_embedding:
      seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    else:
      seqlen, batch = tf.unstack(tf.shape(inputs), num=2)
    with tf.name_scope(p.name):
      node_out = {}
      if not p.input_is_embedding:
        # Will be consumed by all embedding layers.
        ids = py_utils.HasRank(inputs, 2)
        paddings = py_utils.HasShape(paddings, tf.shape(ids))
        node_out[-1] = (ids, paddings)
      else:
        # Will be passed over by the IdentityLayer.
        node_out[-1] = (inputs, paddings)
      # FProp follows the topological order.
      for (i, layer) in enumerate(self.graph_layers):
        nid, fids = self._visit_plan[i]
        need_proj = nid in self._need_father_proj
        # Merge inputs from fathers.
        in_xs = burger_layers.GraphUtils.MergeFatherInputs(
            self, nid, fids, node_out, need_proj)
        _, in_ps = node_out[fids[0]]
        xs, ps = layer.FProp(theta.graph_layers[i], in_xs, in_ps)
        node_out[nid] = (xs, ps)

      xss = []
      # If needed, project and merge outputs
      for o in self._out_nodes:
        xs, ps = node_out[o]
        xs = py_utils.HasShape(xs, [-1, -1, -1])
        out_dim = self._node_out_dims[o]
        if out_dim != p.softmax.input_dim:
          ps_e = tf.expand_dims(ps, 2)
          xs_p = burger_layers.GraphUtils.GetProjection(self, xs, o, 'out',
                                                        ps_e)
          xss.append(xs_p)
        else:
          xss.append(xs)

      # [time, batch, dim]
      xs_out = self.merger_layer.FProp(theta, xss)

      if labels is None:
        # We can only compute the logits here.
        logits = self.softmax.Logits(
            theta=theta.softmax,
            inputs=tf.reshape(xs_out, [seqlen * batch, -1]))
        xent_output = py_utils.NestedMap(
            logits=tf.reshape(logits, [seqlen, batch, -1]))
      elif 'class_ids' in labels:
        xent_output = self.softmax.FProp(
            theta=theta.softmax,
            inputs=xs_out,
            class_weights=labels.class_weights,
            class_ids=labels.class_ids)
      else:
        assert 'class_probabilities' in labels
        xent_output = self.softmax.FProp(
            theta=theta.softmax,
            inputs=xs_out,
            class_weights=labels.class_weights,
            class_probabilities=labels.class_probabilities)
      xent_output.last_hidden = xs_out

      # TODO(yuancao): Output state1 to be consistent with other lm interface.
      return xent_output, None


class FstLm(lingvo_lm_layers.BaseLanguageModel):
  """FST based language model layer."""

  @classmethod
  def Params(cls):
    p = super(FstLm, cls).Params()
    p.Define(
        'word_lm_path', '',
        'Path to an FSA compatible with the nlp/fst library. Inputs and '
        'outputs should both be words.')
    p.Define('word_list_path', '',
             'Path to symbol table compatible with nlp/fst.')
    p.Define(
        'use_sentence_end_token', True,
        'Defines how to signify the end of an utterance to the FST.  A value '
        'of True means the FST has arcs for the end of sentence token </s>. '
        'A value of False means that it uses final weights instead.')
    p.Define(
        'word_end_token', ' ', 'Token used by FST to represent the end '
        'of a word. For grapheme models this is typically " " or '
        '<space>, while for CIP models it is typically <eow>.')
    p.Define('decoding_unit', 'grapheme',
             'The input label unit of the FST: "grapheme" or "wordpiece".')
    p.Define(
        'unit_lm_path', '',
        'Path to FST compatible with the nlp/fst library. Inputs and '
        'outputs should both be graphemes, phonemes, or wordpieces.')
    p.Define(
        'unit_list_path', '',
        'Path to symbol table corresponding to FST stored in '
        '"unit_lm_path". Symbols should, graphemes, phonemes, '
        'or wordpieces.')
    p.Define(
        'wordpiece_model_prefix', '',
        'Path to wordpiece model.  Should include directory and "wpm" '
        'prefix as expected elsewhere.  Only used if decoding_unit is '
        '"wordpiece".')
    p.Define('allow_partial_lookup', False, 'Treat fst as a partial LM with '
             'failure transitions.')
    p.Define(
        'push_speller', True, 'Push fst probabilities to each unit '
        'of the word, rather than having the probability on the last '
        'word.')
    p.Define(
        'failure_transition_token', '', 'If a subtractive backoff arc '
        'cost is included in the FST with this failure_transition_token, '
        'then include the cost for all non matching prefixes.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FstLm, self).__init__(params)
    self._serialized_fst = tf.constant('', dtype=tf.string, name='biasing_fst')
    self._serialized_syms = tf.constant(
        '', dtype=tf.string, name='biasing_syms')

  def zero_state(self, batch_size):
    return py_utils.NestedMap(
        prefixes=tf.zeros(tf.stack([batch_size, 1]), dtype=tf.string))

  def _IdToToken(self, ids):
    p = self.params
    ids = py_utils.HasRank(ids, 1)
    batch = tf.shape(ids)[0]
    ids = tf.expand_dims(ids, -1)  # [batch, 1]
    seqlen = tf.ones([batch], dtype=tf.int32)
    if p.decoding_unit == 'wordpiece':
      return py_x_ops.wpm_id_to_token(
          ids, seqlen, p.wordpiece_model_prefix, undo_segmentation=False)
    elif p.decoding_unit == 'grapheme':
      return py_x_ops.id_to_token(ids, seqlen)

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension."""
    return py_utils.NestedMap(logits=params.vocab_size, last_hidden=0)

  def SetDynamicFst(self, serialized_fst, serialized_syms):
    """Set the serialized_fst and serialized_syms parameters."""

    self._serialized_fst = serialized_fst
    self._serialized_syms = serialized_syms

  def FProp(self, theta, inputs, paddings, state0):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.

    Returns:
      Returns (output, state1), where output is a NestedMap as defined
      and state1 is the next recurrent state. output.pseudo_logits is
      a tensor of [time, batch, vocab], where output.pseudo_logits[i,
      j, k] denotes a log probability if it's non-zero. If it's zero,
      the value should not be considered as a padded position.
    """
    del theta  # No weights.
    p = self.params
    inputs = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(inputs))
    assert state0

    @function.Defun(tf.int32, tf.string, tf.string, tf.string, tf.int32,
                    *[p.dtype] * 2)
    def ForBody(i, serialized_fst, serialized_syms, prefixes, inputs, paddings,
                out_log_probs):
      """For loop body."""
      inputs_i = inputs[i]
      paddings_i = paddings[i]

      # NOTE: if fst_log_prob arranges 'prefixes' to be [time, batch]
      # instead of [batch, time]. In fact, we can move
      # self._IdToToken() out of the for loop to be more efficient.

      # Input tokens must be converted from IDs to tokens.  There is
      # no guarantee that the FST was built with the same vocabulary
      # that is used by the model, so we communicate with it at the
      # token level.
      input_tokens = self._IdToToken(inputs_i)

      # Since states are just prefixes, advancing 'states' by 'input_tokens'
      # amounts to extending each prefix by its corresponding input token.
      prefixes = tf.concat([prefixes, tf.expand_dims(input_tokens, -1)], axis=1)

      # We will compute log probs for all tokens in the vocab.  As
      # above, tokens must be converted from IDs to tokens.
      selection_tokens = self._IdToToken(tf.range(p.vocab_size, dtype=tf.int32))

      # 'prefixes' is of size [batch_size, time] and 'token_selection_tokens' is
      # of size [num_tokens].  'log_probs' is of size [batch_size, num_tokens].
      log_probs = py_x_ops.fst_log_prob(
          prefixes,
          selection_tokens,
          serialized_fst=serialized_fst,
          serialized_syms=serialized_syms,
          word_lm_path=p.word_lm_path,
          word_list_path=p.word_list_path,
          unit_lm_path=p.unit_lm_path,
          unit_list_path=p.unit_list_path,
          use_sentence_end_token=p.use_sentence_end_token,
          word_end_token=p.word_end_token,
          decoding_unit=p.decoding_unit,
          allow_partial_lookup=p.allow_partial_lookup,
          push_speller=p.push_speller,
          wpm_model_path=p.wordpiece_model_prefix + '.model',
          wpm_vocab_path=p.wordpiece_model_prefix + '.vocab',
          failure_transition_token=p.failure_transition_token)

      log_probs = tf.expand_dims(1.0 - paddings_i, axis=-1) * log_probs
      out_log_probs = inplace_ops.alias_inplace_update(
          out_log_probs, [i], tf.expand_dims(log_probs, 0))
      return (serialized_fst, serialized_syms, prefixes, inputs, paddings,
              out_log_probs)

    # The code block only works on cpu.
    with tf.device('/cpu:0'):
      time, batch = tf.unstack(tf.shape(inputs), 2)
      prefixes = state0.prefixes
      log_probs = inplace_ops.empty(
          [time, batch, p.vocab_size], p.dtype, init=True)
      _, _, out_prefixes, _, _, out_log_probs = functional_ops.For(
          start=0,
          limit=time,
          delta=1,
          inputs=[
              self._serialized_fst, self._serialized_syms, prefixes, inputs,
              paddings, log_probs
          ],
          body=ForBody)
      out_prefixes.set_shape(tf.TensorShape([prefixes.shape[0], 1]))
      out_log_probs.set_shape(log_probs.shape)

    output = py_utils.NestedMap(
        pseudo_logits=out_log_probs,
        log_probs=out_log_probs,
        probs=tf.nn.softmax(out_log_probs),
        last_hidden=out_log_probs)
    return (output, py_utils.NestedMap(prefixes=out_prefixes))

  def GetFeedDict(self):
    result = super(FstLm, self).GetFeedDict()
    result.update({
        'biasing_fst': self._serialized_fst,
        'biasing_syms': self._serialized_syms
    })
    return result


class FstRnnLm(lingvo_lm_layers.BaseLanguageModel):
  """A class to interpolate an FST and RNN-based language model.

     Note this class only works during decoding and not training currently.
  """

  @classmethod
  def Params(cls):
    p = super(FstRnnLm, cls).Params()
    p.Define('fst_lm', FstLm.Params(), 'An FST-LM.')
    p.Define('rnn_lm', RnnLm.Params(), 'An RNN-LM.')
    p.Define('fst_weight', 0.0, 'The weight of the fst when interpolating.')
    p.Define('rnn_weight', 1.0, 'The weight of the RNN-LM when interpolating.')
    p.Define(
        'interp_posteriors', True,
        'If set to true, we interpolate the posteriors of the two LMs. '
        'Otherwise we interpolate the log-posteriors.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FstRnnLm, self).__init__(params)
    p = self.params
    assert p.fst_lm.vocab_size == p.rnn_lm.vocab_size
    name = p.name
    self._fst_bias_probs = None
    with tf.variable_scope(name):
      self.CreateChild('fst_lm', p.fst_lm)
      self.CreateChild('rnn_lm', p.rnn_lm)

  def _InterpolateLms(self, rnnlm, fstlm):
    fw = self.params.fst_weight * tf.ones_like(rnnlm.log_probs)
    rw = self.params.rnn_weight * tf.ones_like(rnnlm.log_probs)
    # If the fst score is low, set fw=0 and rw=1. Note min_log_prob is set from
    # here https://cs.corp.google.com/piper///depot/google3/learning/brain/
    # research/babelfish/ops/fst_log_prob_op.h?rcl=197331037&l=34
    min_log_prob = -100000
    fw = tf.where(
        tf.less_equal(fstlm.log_probs, min_log_prob), tf.zeros_like(fw), fw)
    rw = tf.where(
        tf.less_equal(fstlm.log_probs, min_log_prob), tf.ones_like(rw), rw)
    # Interpolate normalized log probabilities
    if self.params.interp_posteriors:
      normalized_interp_probs = rw * tf.nn.softmax(
          rnnlm.log_probs) + fw * tf.nn.softmax(fstlm.log_probs)
      interp_log_probs = tf.log(normalized_interp_probs)
    else:
      interp_logits = rw * rnnlm.log_probs + fw * fstlm.log_probs
      interp_log_probs = tf.nn.log_softmax(interp_logits)
    # Don't touch the hidden state.
    new_xent = py_utils.NestedMap(
        log_probs=interp_log_probs,
        logits=interp_log_probs,
        probs=tf.nn.softmax(interp_log_probs))
    return new_xent

  def SetFstProbs(self, fst_bias_probs):
    self._fst_bias_probs = fst_bias_probs

  def zero_state(self, batch_size):
    fstlm_zero_state = self.fst_lm.zero_state(batch_size)
    rnnlm_zero_state = self.rnn_lm.zero_state(batch_size)
    return py_utils.NestedMap(
        fstlm_state=fstlm_zero_state, rnnlm_state=rnnlm_zero_state)

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension."""
    return py_utils.NestedMap(
        logits=params.rnn_lm.vocab_size,
        last_hidden=params.rnn_lm.softmax.input_dim)

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    # do the fprop with the rnn-lm
    fprop_rnnlm = self.rnn_lm.FProp(theta.rnn_lm, inputs, paddings,
                                    state0.rnnlm_state)

    fst_probs = self._fst_bias_probs
    fst_states = state0.fstlm_state
    if not fst_probs:
      # do the fprop with the fst
      fst_probs, fst_states = self.fst_lm.FProp(theta.fst_lm, inputs, paddings,
                                                state0.fstlm_state)
    # interpolate the outputs
    interpolated_output = self._InterpolateLms(fprop_rnnlm[0], fst_probs)
    new_state = py_utils.NestedMap(
        fstlm_state=fst_states, rnnlm_state=fprop_rnnlm[1])
    return interpolated_output, new_state

  def GetFeedDict(self):
    result = self.rnn_lm.GetFeedDict()
    result.update(self.fst_lm.GetFeedDict())
    return result


class WordLevelFstLm(lingvo_lm_layers.BaseLanguageModel):
  """Word-level FST based language model layer."""

  @classmethod
  def Params(cls):
    p = super(WordLevelFstLm, cls).Params()
    p.Define(
        'word_lm_path', '',
        'Path to an FSA compatible with the nlp/fst library. Inputs and '
        'outputs should both be words.')
    p.Define('word_list_path', '',
             'Path to symbol table compatible with nlp/fst.')
    p.Define('input_is_symbol_index', True,
             'Indicates whether the input is symbol table indices.')
    return p

  def zero_state(self, batch_size):
    return py_utils.NestedMap(
        prefixes=tf.zeros(tf.stack([batch_size, 1]), dtype=tf.string))

  def FProp(self, theta, inputs, paddings, state0=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers. Not being used.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: Not being used.

    Returns:
      Returns (output, state1), where output is a NestedMap as defined
      and state1 is the next recurrent state. output.pseudo_logits is
      a tensor of [time, batch, vocab], where output.pseudo_logits[i,
      j, k] denotes a log probability if it's non-zero. If it's zero,
      the value should not be considered as a padded position.
    """
    del theta  # No weights.
    del state0  # Only replies on 'inputs'.
    p = self.params
    inputs = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(inputs))

    transposed_inputs = tf.transpose(inputs)

    with tf.device('/cpu:0'):
      self._serialized_fst = tf.constant('', dtype=tf.string)
      # 'transposed_inputs' is of size [batch_size, time]. 'log_probs' is of
      # size [batch_size, time, vocab_size].
      log_probs = py_x_ops.word_level_fst_log_prob(
          transposed_inputs,
          self._serialized_fst,
          word_lm_path=p.word_lm_path,
          word_list_path=p.word_list_path,
          input_is_symbol_index=p.input_is_symbol_index)

    # Reshaped to [time, batch, 1]
    paddings = tf.expand_dims(paddings, axis=-1)
    log_probs = (1.0 - paddings) * tf.transpose(log_probs, [1, 0, 2])

    output = py_utils.NestedMap(
        pseudo_logits=log_probs, log_probs=log_probs, last_hidden=log_probs)
    return (output, None)

  def serialized_fst_feed(self):
    return self._serialized_fst


def GatherLogProbsByIds(ids, log_probs):
  """Gathers log_probs corresponding to ids.

  Args:
    ids: int tensor of [time, batch].
    log_probs: tensor of [time, batch, vocab].

  Returns:
    id_log_probs: tensor of [time, batch], where id_log_probs[t, b] =
      log_probs[t, b, ids[t, b]].
  """
  assert ids.shape.ndims == 2, 'shape=%s' % ids.shape
  time = tf.shape(ids)[0]
  batch = tf.shape(ids)[1]
  # [time, batch]: [[0, ..., 0], [1, ..., 1], ..., [time - 1, ..., time - 1]].
  dim0 = tf.tile(tf.expand_dims(tf.range(time), axis=-1), [1, batch])
  # [time, batch]: [[0, 1, ..., batch - 1], ..., [0, ..., batch - 1]].
  dim1 = tf.tile(tf.expand_dims(tf.range(batch), axis=0), [time, 1])
  # [time, batch, 3].
  indices = tf.stack(
      [tf.cast(dim0, dtype=ids.dtype),
       tf.cast(dim1, dtype=ids.dtype), ids],
      axis=-1)
  return tf.gather_nd(log_probs, indices)


class LmWithPhrases(lingvo_lm_layers.BaseLanguageModel):
  """A LM combined with a number of bias phrases."""

  @classmethod
  def Params(cls):
    p = super(LmWithPhrases, cls).Params()
    p.Define('lm', NullLm.Params(), 'Base LM.')
    p.Define('phrase_matching_prior', 0.01,
             'Prior probability of a phrase appearing in an input sequence.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LmWithPhrases, self).__init__(params)
    p = self.params
    assert p.lm.vocab_size == p.vocab_size
    with tf.variable_scope(p.name):
      self.CreateChild('lm', p.lm)
      self.phrases = py_utils.NestedMap(
          ids=tf.zeros([0, 0, 0], dtype=tf.int32, name='phrase_ids'),
          lengths=tf.zeros([0, 0], dtype=tf.int32, name='phrase_lengths'),
          probs=tf.zeros([0, 0], dtype=p.dtype, name='phrase_probs'))

  def SetPhrases(self, phrases):
    """Sets up bias phrases.

    Args:
      phrases: a NestedMap with the follow keys:
        * ids: int32 tensor of [batch, num_phrases, phrase_max_len].
        * lengths: int32 tensor of [batch, num_phrases].
        * probs: tensor of [batch, num_phrases].
    """
    # Sort phrases lexicographically for efficient prefix matching.
    phrases.ids, phrases.lengths, phrases.probs = (
        py_x_ops.sort_phrases(phrases.ids, phrases.lengths, phrases.probs))
    self.phrases = phrases

  def zero_state(self, batch_size):
    p = self.params
    return py_utils.NestedMap(
        lm_state=self.lm.zero_state(batch_size),
        # All tensors in state must have shape [batch, ...].
        lm_log_probs=tf.nn.log_softmax(
            tf.zeros([batch_size, p.lm.vocab_size], dtype=p.dtype)),
        lm_seq_log_probs=tf.zeros([batch_size, 1], dtype=p.dtype),
        inputs=tf.zeros([batch_size, 0], dtype=tf.int32),
        paddings=tf.zeros([batch_size, 0], dtype=p.dtype))

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension."""
    return py_utils.NestedMap(
        logits=params.vocab_size,
        log_probs=params.vocab_size,
        probs=params.vocab_size)

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    """Matches inputs to phrases."""
    p = self.params

    # [time, batch].
    input_len = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    original_seq_len = tf.shape(state0.inputs)[1]
    total_seq_len = original_seq_len + input_len

    # Check input shapes.
    state0.inputs = py_utils.HasShape(state0.inputs, [batch, original_seq_len])
    state0.paddings = py_utils.HasShape(state0.paddings,
                                        [batch, original_seq_len])
    state0.lm_seq_log_probs = py_utils.HasShape(state0.lm_seq_log_probs,
                                                [batch, original_seq_len + 1])

    state1 = py_utils.NestedMap()
    lm_out, state1.lm_state = self.lm.FProp(theta.lm, inputs, paddings,
                                            state0.lm_state)
    lm_log_probs = py_utils.HasShape(lm_out.log_probs,
                                     [input_len, batch, p.vocab_size])

    concat_lm_log_probs = tf.concat(
        [tf.expand_dims(state0.lm_log_probs, 0), lm_log_probs], axis=0)
    lm_seq_log_probs = GatherLogProbsByIds(inputs, concat_lm_log_probs)

    # Append to state0.{input, paddings, lm_seq_log_probs}.
    state1.inputs = tf.concat([state0.inputs, tf.transpose(inputs)], axis=1)
    state1.paddings = tf.concat(
        [state0.paddings, tf.transpose(paddings)], axis=1)
    state1.lm_seq_log_probs = tf.concat(
        [state0.lm_seq_log_probs,
         tf.transpose(lm_seq_log_probs)], axis=1)
    state1.lm_log_probs = concat_lm_log_probs[-1, ...]

    state1.inputs = py_utils.HasShape(state1.inputs, [batch, total_seq_len])
    state1.paddings = py_utils.HasShape(state1.paddings, [batch, total_seq_len])
    state1.lm_seq_log_probs = py_utils.HasShape(state1.lm_seq_log_probs,
                                                [batch, total_seq_len + 1])

    if not p.is_eval:
      output = py_utils.NestedMap(log_probs=lm_log_probs, logits=lm_log_probs)
      return output, state1

    recurrent_theta = py_utils.NestedMap(
        phrases=self.phrases,
        inputs=state1.inputs,
        paddings=state1.paddings,
        cum_lm_seq_log_probs=tf.cumsum(state1.lm_seq_log_probs, axis=1))
    recurrent_state0 = py_utils.NestedMap(
        step=original_seq_len,
        probs=tf.zeros([batch, p.vocab_size]),
        log_probs=tf.zeros([batch, p.vocab_size]))
    recurrent_inputs = py_utils.NestedMap(lm_log_probs=lm_log_probs)

    def Step(theta, step_state0, inputs):
      """FProp for one step.

      Args:
        theta: A NestedMap with 'phrases', 'inputs', 'paddings', and
          'cum_lm_seq_log_probs', where phrases, inputs, and paddings come from
          FProp() args and cum_lm_seq_log_probs[b, k] represents the total LM
          log_prob of the first k tokens for the b'th sequence.
        step_state0: A NestedMap with 'step', 'prob', 'log_prob', where 'prob'
          and 'log_prob' are output probabilities/log_probs (after interpolation
          with phrase matching) and have shape [batch, vocab].
        inputs: A NestedMap with 'lm_log_prob', representing LM's output
          log_probs for the time step, of shape [batch, vocab].

      Returns:
        (step_state1, <empty NestedMap>).
      """
      theta.cum_lm_seq_log_probs.set_shape([None, None])  # [batch, time]
      inputs.lm_log_probs.set_shape([None, p.vocab_size])  # [batch, vocab]

      # Step represents the time step the current input ids, e.g., when step is
      # 0, seq_len is 1 (including the current input step).
      seq_len = step_state0.step + 1
      seq_ids = theta.inputs[:, :seq_len]
      seq_paddings = theta.paddings[:, :seq_len]
      match_lens, prefix_probs, next_probs = (
          py_x_ops.find_phrase_prefix(
              seq_ids,
              # Sequence lengths.
              tf.cast(tf.reduce_sum(1 - seq_paddings, axis=1), tf.int32),
              theta.phrases.ids,
              theta.phrases.lengths,
              theta.phrases.probs,
              vocab_size=p.vocab_size))

      # Compute a linear interpolation between next_probs and
      # tf.exp(inputs.lm_log_probs) with Bayesian interpolation weights.
      #
      # First, we compute probabilities of the current sequence under LM +
      # phrases vs. under LM alone. These probabilities are combined with
      # priors to compute interpolation weights.
      base_phrase_log_prob = tf.log(p.phrase_matching_prior)
      base_no_phrase_log_prob = tf.log(1 - p.phrase_matching_prior)

      # [batch].
      seq_prefix_log_probs = tf.gather_nd(
          theta.cum_lm_seq_log_probs,
          tf.stack([tf.range(batch), seq_len - match_lens], axis=1))
      # [batch].
      phrase_prefix_log_probs = tf.log(1e-9 + prefix_probs)
      # [batch]. log_prob of sequences with matching phrases.
      phrase_logits = (
          base_phrase_log_prob + seq_prefix_log_probs + phrase_prefix_log_probs)
      # [batch]. log_prob of sequences ingoring matching phrases.
      no_phrase_logits = (
          base_no_phrase_log_prob + theta.cum_lm_seq_log_probs[:, -1])
      # [2, batch].
      logits = tf.stack([phrase_logits, no_phrase_logits], axis=0)
      assert logits.shape.ndims == 2, 'shape=%s' % logits.shape
      # [2, batch, 1].
      weights = tf.expand_dims(tf.nn.softmax(logits, axis=0), axis=-1)

      # Interpolation of next token probabilities by weights.
      # [2, batch, vocab].
      stacked_probs = tf.stack(
          [next_probs, tf.exp(inputs.lm_log_probs)], axis=0)
      step_state1 = py_utils.NestedMap(step=step_state0.step + 1)
      # [batch, vocab].
      step_state1.probs = tf.reduce_sum(stacked_probs * weights, axis=0)
      step_state1.log_probs = tf.log(1e-9 + step_state1.probs)
      return step_state1, py_utils.NestedMap()

    acc_state, _ = recurrent.Recurrent(
        recurrent_theta,
        recurrent_state0,
        recurrent_inputs,
        Step,
        allow_implicit_capture=p.allow_implicit_capture)
    output = py_utils.NestedMap(
        probs=acc_state.probs,
        log_probs=acc_state.log_probs,
        logits=acc_state.log_probs)
    return output, state1

  def GetFeedDict(self):
    result = super(LmWithPhrases, self).GetFeedDict()
    result.update({
        'phrase_ids': self.phrases.ids,
        'phrase_lengths': self.phrases.lengths,
        'phrase_probs': self.phrases.probs,
    })
    return result


class SinbadMaxEntLm(lingvo_lm_layers.BaseLanguageModel):
  """Sinbad Maximum-Entropy Language model."""

  @classmethod
  def Params(cls):
    p = super(SinbadMaxEntLm, cls).Params()
    p.Define('sinbad_config_path', '',
             'Path to the Sinbad configuration to load.')
    p.Define(
        'babelfish_vocabulary_path', '',
        'Path to the Babelfish vocabulary, used to translate between '
        'Sinbad and Babelfish token ids.  Must be in the same TSV format '
        'as a Sinbad vocabulary.')
    return p

  @base_layer.initializer
  def __init__(self, params, compute_logits=None):
    """Create a Sinbad rescoring layer.

    Args:
      params: Rescorer parameters.  See Params() for details.
      compute_logits: Test-only optional function to use to compute logits.  If
        not specified, sinbad_ops.sinbad_compute_logits() will be used.  If
        specified, this function's signature must match that of
        sinbad_ops.sinbad_compute_logits().  Note that, when this parameter
        is specified, the Sinbad handle and label translation resources will
        not be created.
    """

    super(SinbadMaxEntLm, self).__init__(params)

    sinbad_config_path = self.params.sinbad_config_path
    with tf.gfile.FastGFile(sinbad_config_path, 'r') as sinbad_config_file:
      sinbad_config = text_format.ParseLines(sinbad_config_file,
                                             sinbad_pb2.SinbadConfig())
      self._max_context_length = sinbad_config.fingerprint_window

    # If the user specified a compute_logits function to use in unit testing,
    # use that instead of sinbad_ops.sinbad_compute_logits().
    if compute_logits:
      self._compute_logits = compute_logits
      # When a custom compute_logits function is specified, we don't create the
      # remaining resources.  However, we do explicitly set them to None to
      # signal that the object was initialized properly.
      self._sinbad = None
      self._external_to_sinbad_label_translation_resource = None
      self._sinbad_to_external_label_translation_resource = None
      return

    self._compute_logits = sinbad_ops.sinbad_compute_logits
    sinbad_symbol_table_path = os.path.join(
        os.path.dirname(self.params.sinbad_config_path),
        sinbad_config.vocabulary_file)

    # Get a handle to the Sinbad predictor and create translation tables between
    # Babelfish's vocabulary and the Sinbad model's vocabulary.
    self._sinbad = sinbad_ops.sinbad_manager_resource_handle(sinbad_config_path)
    self._external_to_sinbad_label_translation_resource = (
        sinbad_ops.label_translation_resource_handle(
            source_symbol_table_path=self.params.babelfish_vocabulary_path,
            target_symbol_table_path=sinbad_symbol_table_path))
    self._sinbad_to_external_label_translation_resource = (
        sinbad_ops.label_translation_resource_handle(
            source_symbol_table_path=sinbad_symbol_table_path,
            target_symbol_table_path=self.params.babelfish_vocabulary_path))

  def zero_state(self, batch_size):
    """Create the initial recurrent state.

    Args:
      batch_size: int, the number of hypotheses to score.

    Returns:
      A NestedMap containing two entries:
        prev_ids: An int64 Tensor of shape
          [batch size, self._max_context_length] which will be populated with
          the context of token ids preceding each input in the batch, plus
          leading padding if the context is shorter than
          self._max_context_length.
        padding_length: int32 scalar, The number of padding symbols in prev_ids
          before the actual context.  Initially, prev_ids contains only
          padding; the initial padding length is _max_context_length + 1 because
          we also want to consider the first input symbol, the BOS, as part of
          the padding, since Sinbad models BOS implicitly and so we should
          not include the BOS in the contexts we pass to Sinbad.
    """

    return py_utils.NestedMap(
        prev_ids=tf.zeros(
            [batch_size, self._max_context_length], dtype=tf.int64),
        padding_length=tf.constant(
            self._max_context_length + 1, dtype=tf.int32),
    )

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension.

    Args:
      params: Params for this layer.

    Returns:
      output_dims: A NestedMap with fields.
        logits: a python int. the vocab size.
    """

    return py_utils.NestedMap(logits=params.vocab_size)

  def _ComputeSingleHypAllPositionLogits(self, context):
    """Return the logits for all positions of the sequence.

    Args:
      context: Tensor of shape [batch, time], where each item in the batch
        contains the previous token ids (without the BOS symbol), along with
        a padding symbol in the position to be predicted.

    Returns:
      Tensor of shape [batch, time + 1, vocabulary size] containing the logits
      for each token in the sequence, plus the position to be predicted.
    """

    # Pad the input to provide a position for the logits of the token to be
    # predicted.
    context = tf.pad(context, [[0, 1]])
    return self._compute_logits(
        self._sinbad,
        self._external_to_sinbad_label_translation_resource,
        self._sinbad_to_external_label_translation_resource,
        context,
        self.params.vocab_size,
        check_vocabulary=True)

  def _UpdateState(self, state0, batch_major_inputs):
    """Updates the recurrent state to track the input just seen.

    Args:
      state0: The current recurrent state.
      batch_major_inputs: Tensor of shape [batch, time], containing the input
        to update the state with.

    Returns:
      The new recurrent state.
    """

    full_prev_ids = tf.concat([state0.prev_ids, batch_major_inputs], axis=1)
    input_length = tf.shape(batch_major_inputs)[1]
    state1 = py_utils.NestedMap(
        # Maintain a window of length self._max_context_length on the previous
        # ids seen in all steps, i.e., all previous inputs plus this one.
        prev_ids=full_prev_ids[:, -self._max_context_length:],
        padding_length=tf.maximum(0, state0.padding_length - input_length),
    )
    state1.Transform(tf.stop_gradient)

    return state1

  def _ComputeCrossEntropy(self, logits, labels):
    """Computes cross-entropy between logits and labels.

    Args:
      logits: Tensor of shape [time, batch, vocab_size].
      labels: optional NestedMap, containing labels as either class_ids or
        class_probabilities.

    Returns:
      A NestedMap as defined by SoftmaxLayer's return value.
    """

    xent_output = py_utils.NestedMap(
        logits=logits,
        last_hidden=logits,
    )

    if labels:
      if 'class_ids' in labels:
        per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels.class_ids, logits=logits)
      else:
        assert 'class_probabilities' in labels
        per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels.class_probabilities, logits=logits)

      total_xent = tf.reduce_sum(per_example_xent * labels.class_weights)
      total_weights = tf.reduce_sum(labels.class_weights)
      xent_output.update({
          'log_probs': tf.nn.log_softmax(logits),
          'per_example_argmax': py_utils.ArgMax(logits),
          'per_example_xent': per_example_xent,
          'per_example_weight': labels.class_weights,
          'total_xent': total_xent,
          'total_weight': total_weights,
          'avg_xent': total_xent / total_weights,
      })

    xent_output.Transform(tf.stop_gradient)
    return xent_output

  def _ComputeLogitsAndCrossEntropy(self,
                                    batch_major_inputs,
                                    state,
                                    labels=None):
    """Computes logits and, if labels are given, cross-entropy.

    Args:
      batch_major_inputs: Tensor of shape [batch, time], input symbols.
      state: NestedMap, current recurrent state.
      labels: optional NestedMap, containing labels as either class_ids or
        class_probabilities.

    Returns:
      A NestedMap as defined by SoftmaxLayer's return value.
    """

    full_prev_ids = tf.concat([state.prev_ids, batch_major_inputs], axis=1)
    # Note that, since the initial padding length is _max_context_length + 1,
    # this will remove the initial BOS from the first input when creating the
    # context.  This is necessary because Sinbad models BOS implicitly.
    contexts = full_prev_ids[:, state.padding_length:]
    batch_major_non_padding_logits = tf.map_fn(
        self._ComputeSingleHypAllPositionLogits,
        contexts,
        dtype=tf.float32,
        back_prop=False)

    # _ComputeSingleHypAllPositionLogits() computes the logits at all positions,
    # but we only care about the logits for the inputs, not the previous ids in
    # the context.
    #
    # TODO(b/75969601): For efficiency, we should only compute the logits for
    # the inputs, not the full context including the previous ids.  Consider
    # adding a "constant prefix" to the logit computation.
    input_length = tf.shape(batch_major_inputs)[1]
    batch_major_logits_for_input = (
        batch_major_non_padding_logits[:, -input_length:, :])

    # Transpose back to [time, batch, vocab_size] order.
    logits = tf.transpose(batch_major_logits_for_input, [1, 0, 2])

    return self._ComputeCrossEntropy(logits, labels)

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step using the Sinbad MaxEnt predictor.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.  Unused.
      inputs: a tensor of shape [batch].
      paddings: a 0/1 tensor of shape [batch].
      state0: A NestedMap containing the initial recurrent state.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      output: A NestedMap with fields.
        logits: [batch, vocab_size].
        log_probs: [batch, vocab_size].
        last_hidden: [batch, dims]; since this has no meaning for MaxEnt,
          we return None.
      state1: A NestedMap containing the new recurrent state.
    """

    del theta

    inputs = tf.cast(py_utils.HasRank(inputs, 1), dtype=tf.int64)
    inputs_with_time_dimension = tf.expand_dims(inputs, axis=1)
    xent_output = self._ComputeLogitsAndCrossEntropy(inputs_with_time_dimension,
                                                     state0)
    # Truncate the returned logits to only include those of the last time step.
    # The logits have shape [batch, vocab_size].
    logits = xent_output.logits[-1, :, :]
    state1 = self._UpdateState(state0, inputs_with_time_dimension)
    # Note that the logits returned by Sinbad are already normalized logprobs:
    # http://google3/speech/languagemodel/sinbad/hierarchical_predictor.cc?l=248&rcl=188222827.
    return (py_utils.NestedMap(
        logits=logits, log_probs=logits, last_hidden=None), state1)

  def FProp(self, theta, inputs, paddings, state0, labels=None):
    """Computes xent loss given the language model inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.  Unused.
      inputs: a tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.

    Returns:
      (xent_output, state1). xent_output is a NestedMap as defined by
      SoftmaxLayer's return value and state1 is the next recurrent
      state.
    """

    inputs = py_utils.HasRank(inputs, 2)
    batch_major_inputs = tf.cast(tf.transpose(inputs), dtype=tf.int64)
    xent_output = self._ComputeLogitsAndCrossEntropy(batch_major_inputs, state0,
                                                     labels)
    state1 = self._UpdateState(state0, batch_major_inputs)

    return xent_output, state1


class HRREmbeddingLayer(base_layer.LayerBase):
  """HRR embedding layer"""

  @classmethod
  def Params(cls):
    p = super(HRREmbeddingLayer, cls).Params()
    p.Define('embedding_dim', 0, 'Embedding size')
    p.Define('num_roles', 0, 'Number of different roles (n)')
    # TODO(jmluo)
    # might want to use different m values for different roles.
    p.Define('num_fillers_per_role', 20,
             'Number of different fillers for each role (m)')
    p.Define('e_l', layers.EmbeddingLayer.Params(), 'Lexicalized embedding')
    # note that s is used num_roles times
    p.Define('s', layers.EmbeddingLayer.Params(), 'Signature embedding')
    # p.Define('rs', layers.EmbeddingLayer.Params(), 'Role signature')
    p.Define('mode', 'basic', 'Modes')
    p.Define('merge', False, 'Flag to merge all collections of filler matrices into a big one')
    # TODO(jmluo)
    p.Define('vocab_size', 0, 'Vocabulary size')
    p.Define('actual_shards', -1, 'Actual number of shards used. This should not be specified, but computed during __init__ call')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(HRREmbeddingLayer, self).__init__(params)
    p = self.params
    assert p.embedding_dim > 0
    assert p.num_roles > 0
    assert p.num_fillers_per_role > 0
    assert p.vocab_size > 0
    assert p.e_l.vocab_size == p.vocab_size == p.s.vocab_size
    assert p.e_l.embedding_dim == p.embedding_dim
    assert p.s.embedding_dim == p.num_fillers_per_role * p.num_roles
    assert p.actual_shards == p.e_l.actual_shards == p.s.actual_shards
    assert p.mode in ['basic', 'rs', 'dec_only']
    if p.merge:
      assert p.mode == 'rs', 'Other modes not supported yet'

    r_pc = py_utils.WeightParams(
        shape=[p.num_roles, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    F_pc = py_utils.WeightParams(
        shape=[p.num_roles, p.num_fillers_per_role, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      # TODO(jmluo) disabled for now
      # self.CreateChild('e_l', p.e_l)

      if p.mode == 'rs':
        rr_pc = py_utils.WeightParams(
            shape=[p.num_roles, p.embedding_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        rs = p.s.Copy()
        rs.embedding_dim = 2 * p.num_roles
        rs.name = 'rs'
        rs.params_init = py_utils.WeightInit.PositiveUniform()
        # const = [[1., 0.], [0., 1.]]
        # const = [const] * rs.vocab_size
        # rs.params_init = py_utils.WeightInit.Constant(scale=const)
        # rs.trainable = False

        self.CreateChild('rs', rs)
        self.CreateVariable('rR', rr_pc)
        self.CreateChild('s', p.s)
        self.CreateVariable('F', F_pc)
      elif p.mode == 'basic':
        self.CreateChild('s', p.s)
        self.CreateVariable('r', r_pc)
        self.CreateVariable('F', F_pc)
      else:
        self.CreateChild('e_l', p.e_l)
        self.CreateVariable('r', r_pc)


  def _circular_conv(self, a, b):
    with tf.name_scope('circular_conv'):
      a_fft = tf.fft(tf.complex(a, 0.0))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  def _circular_corr(self, a, b):
    with tf.name_scope('circular_corr'):
      a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  def decode(self, x, r):
    # r_weight: nr x d
    # x: ? x d
    with tf.name_scope('HRR_decode'):
      res = self._circular_corr(r, x)
    return res

  @staticmethod
  def static_circular_conv(a, b):
    with tf.name_scope('static_circular_conv'):
      a_fft = tf.fft(tf.complex(a, 0.0))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  @staticmethod
  def static_circular_corr(a, b):
    with tf.name_scope('static_circular_corr'):
      a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  @staticmethod
  def static_decode(x, r):
    # r_weight: nr x d
    # x: ? x d
    with tf.name_scope('static_HRR_decode'):
      res = HRREmbeddingLayer.static_circular_corr(r, x)
    return res

  def EmbLookup(self, theta, ids, role_anneal=False):
    """Looks up embedding vectors for ids.

    Args:
      theta: Named tuple with the weight matrix for the embedding.
      ids: A rank-N int32 tensor.

    Returns:
      embs: A rank-(N+1) params.dtype tensor. embs[indices, :] is the
        embedding vector for ids[indices].
    """
    p = self.params

    with tf.name_scope('HRR_emb_lookup'):
      emb_weights = self._Emb2Weight(theta, role_anneal=role_anneal)

      # e_l = self.e_l.EmbLookup(theta.e_l, ids)  # size: l x bs x d
      # s_list = list()
      # for i in xrange(self.params.num_roles):
      #   layer = getattr(self, 's%d' % i)
      #   layer_theta = getattr(theta, 's%d' % i)
      #   s = layer.EmbLookup(layer_theta, ids)
      #   # softmax
      #   # s = tf.exp(tf.nn.log_softmax(s, axis=2)) # l x bs x nf
      #   # sigmoid
      #   # s = tf.tanh(s)
      #   s_list.append(s)
      # s_cat = tf.stack(s_list, axis=2)  # size: l x bs x nr x nf

      # old code
  #     s_cat = self.s.EmbLookup(theta.s, ids)  # size: l x bs x nr*nf
  #     shape = tf.shape(s_cat)
  #     l = shape[0]
  #     bs = shape[1]
  #     s_cat = tf.reshape(s_cat, [l, bs, p.num_roles, p.num_fillers_per_role])
  #     f = tf.reduce_sum(
  #         tf.expand_dims(s_cat, 4) * theta.F, axis=3)  # size: l x bs x nr x d
  #
  #
  #
  #     r_conv_f = self._circular_conv(theta.r, f)  # size: l x bs x nr x d
  #     # emb = tf.reduce_sum(r_conv_f, axis=2) + e_l
  #     emb = tf.reduce_sum(r_conv_f, axis=2)  # size: l x bs x d

      emb = tf.nn.embedding_lookup(emb_weights.e, ids, partition_strategy=p.s.partition_strategy)
      s_cat = None

    # distribution constraint
    # mean, variance = tf.nn.moments(emb, axes=[2]) # size: l x bs, l x bs
    # mean = tf.expand_dims(mean, axis=2)
    # variance = tf.expand_dims(variance, axis=2)
    # d = tf.shape(emb)[2]
    # (emb - mean) / tf.sqrt(variance * d)


    return emb, s_cat, emb_weights

  def _Emb2Weight(self, theta, role_anneal=False):
    p = self.params
    e_weights = list()
    rf_weights = list()
    f_weights = list()

    if p.mode == 'rs':
      bases = self._circular_conv(tf.expand_dims(theta.rR, axis=1), theta.F) # size: nr x nf x d
      for rs_shard, s_shard in zip(theta.rs.wm, theta.s.wm):
        # old broken code
        # if p.merge:
        #   F_weights = theta.F[0]
        #   s_shard = tf.reshape(s_shard, [-1, p.num_fillers_per_role])
        #   hid_f_shard = tf.matmul(s_shard, F_weights)
        #   hid_f_shard = tf.reshape(hid_f_shard, [-1, p.num_roles, p.embedding_dim])
        #   rs_shard = tf.reshape(rs_shard, [-1, 2])
        #   hid_r_shard = tf.matmul(rs_shard, theta.rR)
        #   hid_r_shard = tf.reshape(hid_r_shard, [-1, p.num_roles, p.embedding_dim])
        # else:
        #   s_shard = tf.reshape(s_shard, [-1, p.num_roles, p.num_fillers_per_role])
        #   rs_shard = tf.reshape(rs_shard, [-1, p.num_roles, 2])
        #   hid_f_shard_list = list()
        #   hid_r_shard_list = list()
        #   for role_ind in xrange(p.num_roles):
        #     F_weights =
        #     hid_f_shard_i = tf.matmul(s_shard[:, role_ind], theta.F[role_ind]) # size: V/n_shards x d
        #     hid_r_shard_i = tf.matmul(rs_shard[:, role_ind], theta.rR) # size: V/n_shards x d
        #     hid_f_shard_list.append(hid_f_shard_i)
        #     hid_r_shard_list.append(hid_r_shard_i)
        #   hid_f_shard = tf.stack(hid_f_shard_list, axis=1) # size: V/n_shards x nr x d
        #   hid_r_shard = tf.stack(hid_r_shard_list, axis=1)
        rs_shard = tf.reshape(rs_shard, [-1, p.num_roles, 2])
        s_shard = tf.reshape(s_shard, [-1, p.num_roles, p.num_fillers_per_role])
        coeffs = tf.matmul(tf.transpose(rs_shard, perm=[0, 2, 1]), s_shard) # size: V/n_shards x nr x nf
        coeffs_t = tf.transpose(coeffs, [1, 0, 2])
        rf_shard = tf.matmul(coeffs_t, bases) # size: nr x V/n_shards x d
        e_shard = tf.reduce_sum(rf_shard, axis=0)
        # old
        # rf_shard = self._circular_conv(hid_r_shard, hid_f_shard)
        # e_shard = tf.reduce_sum(rf_shard, axis=1) # size: V/n_shards x d
        e_weights.append(e_shard)
        rf_weights.append(rf_shard)
        # real f shard
        f_shard = self._circular_corr(theta.rR, tf.expand_dims(e_shard, axis=1))
        f_weights.append(f_shard)
        # f_weights.append(hid_f_shard)
        r_weights = theta.rR
    elif p.mode == 'basic':
      for s_shard in theta.s.wm:
        s_shard = tf.reshape(s_shard, [-1, p.num_roles, p.num_fillers_per_role])
        f_shard_list = list()
        for role_ind in xrange(p.num_roles):
          f_shard_i = tf.matmul(s_shard[:, role_ind], theta.F[role_ind]) # size: V/n_shards x d
          f_shard_list.append(f_shard_i)
        f_shard = tf.stack(f_shard_list, axis=1) # size: V/n_shards x nr x d
        # TODO(jmluo) revert this
        # if role_anneal:
        #   prob_1 = tf.ones(shape=tf.shape(f_shard_list[0]))
        #   global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
        #   temperature = tf.minimum(tf.constant(3000.0), global_step) / 3000
        #   probs = tf.stack([prob_1, prob_1 * temperature], axis=1)
        #   f_shard = f_shard * probs

        # f_shard = tf.transpose(tf.matmul(tf.transpose(s_shard, perm=[1, 0, 2]), theta.F), perm=[1, 0, 2]) # |V|/n_shards x nr x d
        # f_shard = tf.reduce_sum(s_shard * theta.F, axis=2) # size: V/n_shards x nr x d
        rf_shard = self._circular_conv(theta.r, f_shard)
        e_shard = tf.reduce_sum(rf_shard, axis=1)
        e_weights.append(e_shard)
        rf_weights.append(rf_shard)
        f_weights.append(f_shard)
        # noisy_f_shard = self._circular_corr(theta.r, tf.expand_dims(e_shard, axis=1))
        # f_weights.append(noisy_f_shard)
        r_weights = theta.r
    else:
      e_weights = list()
      f_weights = list()
      r_weights = theta.r
      for e_shard in theta.e_l.wm:
        e_weights.append(e_shard)
        e_shard = tf.reshape(e_shard, [-1, 1, p.embedding_dim])
        f_shard = self._circular_corr(theta.r, e_shard) # size: V/n_shard x nr x d
        f_weights.append(f_shard)

    # NOTE all following weights are sharded along the |V| axis, except r_weights which are
    # not sharded.
    return py_utils.NestedMap(e=e_weights,
                              # rf=rf_weights,
                              r=r_weights,
                              f=f_weights)
