"""Babelfish layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from google3.third_party.tensorflow.python.framework import function
from google3.third_party.tensorflow.python.ops import array_ops
from google3.third_party.tensorflow.python.ops import inplace_ops
from google3.third_party.tensorflow.python.ops import nn_ops
from google3.third_party.tensorflow_lingvo.core import base_layer
from google3.third_party.tensorflow_lingvo.core import layers as lingvo_layers
from google3.third_party.tensorflow.python.ops import candidate_sampling_ops
from google3.third_party.tensorflow_lingvo.core import summary_utils
from google3.learning.brain.google.python.ops import math_ops as internal_math_ops
from google3.learning.brain.models.translation.ops import py_x_ops as nmt_py_x_ops
from google3.learning.brain.research.babelfish import py_utils
from google3.learning.brain.research.babelfish import recurrent
from google3.learning.brain.research.babelfish.ops import py_x_ops

assert_shape_match = py_utils.assert_shape_match
assert_equal = py_utils.assert_equal

# pyformat: disable
# pylint: disable=invalid-name
LOG_SCALE_CLAMP_BOUND = lingvo_layers.LOG_SCALE_CLAMP_BOUND

FPropDtype = lingvo_layers.FPropDtype

IdentityLayer = lingvo_layers.IdentityLayer
BatchNormLayer = lingvo_layers.BatchNormLayer
ConvLayer = lingvo_layers.ConvLayer
DropoutLayer = lingvo_layers.DropoutLayer
DeterministicDropoutLayer = lingvo_layers.DeterministicDropoutLayer
ProjectionLayer = lingvo_layers.ProjectionLayer
FCLayer = lingvo_layers.FCLayer
LayerNorm = lingvo_layers.LayerNorm
PoolingLayer = lingvo_layers.PoolingLayer

EmbeddingLayer = lingvo_layers.EmbeddingLayer
SimpleEmbeddingLayer = lingvo_layers.SimpleEmbeddingLayer
PositionalEmbeddingLayer = lingvo_layers.PositionalEmbeddingLayer

SoftmaxLayer = lingvo_layers.SoftmaxLayer

FeedForwardNet = lingvo_layers.FeedForwardNet

ConvSetLayer = lingvo_layers.ConvSetLayer
UniformLabelSmoother = lingvo_layers.UniformLabelSmoother
HighwaySkipLayer = lingvo_layers.HighwaySkipLayer
GradNormTracker = lingvo_layers.GradNormTracker
WeightedSumLayer = lingvo_layers.WeightedSumLayer
# pylint: enable=invalid-name
# pyformat: enable



# TODO(jmluo) HACK
def sampled_softmax_logits(weights,
                                    biases,
                                    labels,
                                    inputs,
                                    num_sampled,
                                    num_classes,
                                    sampled_values,
                                    num_true=1,
                                    remove_accidental_hits=True,
                                    partition_strategy='div',
                                    name='sampled_softmax_logits',
                                    subtract_log_q=True,
                                    seed=None):
  logits, labels = tf.nn.compute_sampled_logits(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=subtract_log_q,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name,
      seed=seed)
  labels = tf.stop_gradient(labels, name="labels_stop_gradient")
  return logits, labels


class PoolingOverTime(base_layer.LayerBase):
  """Max Pooling applied along the time axis."""

  @classmethod
  def Params(cls):
    p = super(PoolingOverTime, cls).Params()
    p.Define('pooling', 'MAX', 'Pooling to use. Options are MAX, MEAN, NONE.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(PoolingOverTime, self).__init__(params)
    p = self.params
    assert p.name
    assert p.pooling == 'NONE' or p.pooling in ['MAX', 'MEAN']

  def FProp(self, theta, inputs, paddings=None):
    """Apply the pooling methods to inputs along the time axis.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [time, batch, input_dim].
      paddings: The paddings tensor.  Shaped [time, batch, 1], where all but the
      last dimension match.
    Returns:
      Output after applying pooling layer along the time dimension. Shaped
      [batch, dim].
    """
    if paddings is None:
      paddings = tf.zeros(
          tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)
    p = self.params
    inputs = py_utils.with_dependencies([
        assert_shape_match(
            inplace_ops.inplace_update(tf.shape(inputs), [-1], [-1]),
            inplace_ops.inplace_update(tf.shape(paddings), [-1], [-1]))
    ], inputs)
    with tf.name_scope(p.name):
      if p.pooling != 'NONE':
        mask = 1.0 - paddings
        inputs *= mask  # set zero in the padding part
        if p.pooling == 'MAX':
          min_input = tf.reduce_min(inputs, axis=0)
          inputs += min_input * paddings
          out = tf.reduce_max(inputs, axis=0)
        elif p.pooling == 'MEAN':
          out = tf.reduce_sum(inputs, axis=0)
          seq_len = tf.reduce_sum(mask, axis=0)
          out /= (seq_len + 1e-12)
      return out


class StackingOverTime(base_layer.LayerBase):
  """Stacking applied along the time axis.

     At each time step of an input sequence, elements are stacked over the
     window of ('left_context' + 1 + 'right_context') steps around the current
     time step. Zeros will be padded to the left or right of the sequence for
     elements around the boundaries. Finally the stacked outputs are emitted
     once every 'stride' steps.

     E.g. if an input sequence is: [4], [1], [9], [3], [5], [2], [8]
     left_context = 1, right_context = 1, stride = 3,
     then the output sequence would be: [0, 4, 1], [9, 3, 5], [2, 8, 0]

     Note that this layer only performs tensor transformation, so there are no
     learnable parameters.
  """

  @classmethod
  def Params(cls):
    p = super(StackingOverTime, cls).Params()
    p.Define('left_context', 0,
             'Number of time steps to stack on the left to the central step.')
    p.Define('right_context', 0,
             'Number of time steps to stack on the right to the central step.')
    p.Define('stride', 1, 'The stride for emitting the stacked output.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(StackingOverTime, self).__init__(params)
    p = self.params
    assert p.name
    assert p.left_context >= 0
    assert p.right_context >= 0
    assert p.stride >= 1

  @property
  def window_size(self):
    """Returns the stacking window size.

    The output dimension will be window_size * the input dimension.

    Returns:
      Window size.
    """
    p = self.params
    return p.left_context + p.right_context + 1

  def _ApplyStack(self, inputs, pad_value=0.0):
    """The core function to apply the stacking to inputs.

    Args:
      inputs: [batch, time, depth].
      pad_value: the padding value for left/right context.
    Returns:
      out: [batch, ceil(time / stride), depth * stacking_window_length].
    """
    p = self.params
    if p.left_context == 0 and p.right_context == 0:
      out = inputs
    else:
      batch = tf.shape(inputs)[0]
      depth = tf.shape(inputs)[2]
      # Add zero paddings to the left and right of the input sequence.
      padded_inputs = inputs
      if p.left_context > 0:
        left_padding = tf.cast(
            tf.fill([batch, p.left_context, depth], pad_value), inputs.dtype)
        padded_inputs = tf.concat([left_padding, padded_inputs], 1)
      if p.right_context > 0:
        right_padding = tf.cast(
            tf.fill([batch, p.right_context, depth], pad_value), inputs.dtype)
        padded_inputs = tf.concat([padded_inputs, right_padding], 1)

      # Original sequence length before padding.
      inputs_max_len = tf.shape(inputs)[1]
      # Make p.stacking copies of the padded sequence with the original sequence
      # length, where each copy is offset by 1 time step.
      pieces = []
      stacking_window_len = p.left_context + 1 + p.right_context
      for i in range(stacking_window_len):
        pieces.append(padded_inputs[:, i:i + inputs_max_len, :])
      # Apply stacking.
      out = tf.concat(pieces, 2)

    # Apply striding.
    out = out[:, ::p.stride, :]
    return out

  def FProp(self, inputs, paddings=None):
    """Apply the stacking to inputs along the time axis.

    Args:
      inputs: The inputs tensor. It is expected to be of shape [batch,
        time, feature].
      paddings: The paddings tensor. It is expected to be of shape [batch,
        time, 1], where all but the last dimension match those of inputs.
        Each value is 0 or 1 indicating whether a time step of a sequence
        is padded in the inputs to reach the max length in the batch.
    Returns:
      (outputs, out_paddings) pair.
        outputs is of shape [batch, ceil(time / stride), feature * stacking].
        out_paddings is of shape [batch, ceil(time / stride), 1].
    """
    if paddings is None:
      paddings = tf.zeros(
          tf.concat([tf.shape(inputs)[:-1], [1]], 0), dtype=inputs.dtype)
    inputs = py_utils.with_dependencies(
        [
            # Checks the inputs shape has 3 dimensions.
            assert_shape_match(tf.shape(inputs), [-1, -1, -1]),
            # Checks the paddings shape has 3 dimensions, and the last one is 1.
            assert_shape_match(tf.shape(paddings), [-1, -1, 1]),
            # Checks the first two dimensions of inputs and paddings shapes match.  # pylint: disable=line-too-long
            assert_shape_match(tf.shape(inputs)[:-1],
                               tf.shape(paddings)[:-1])
        ],
        inputs)
    p = self.params
    with tf.name_scope(p.name):
      outputs = self._ApplyStack(inputs)

      # Stack the padding values with the same context and stride parameters.
      # Then take the minimum padding values within each stacking window, since
      # an output time step becomes a padded one only if all of the underlying
      # stacked steps are padded ones.
      out_paddings = self._ApplyStack(paddings, pad_value=1.0)
      out_paddings = tf.reduce_min(out_paddings, axis=2, keep_dims=True)

      return outputs, out_paddings


class GradientReversalLayer(base_layer.LayerBase):
  """Gradient Reversal layer.

  Pass the input itself in forward propagation.
  Flip the gradients in back propagation.
  Paper: https://arxiv.org/pdf/1409.7495.pdf
  """

  @classmethod
  def Params(cls):
    p = super(GradientReversalLayer, cls).Params()
    p.Define(
        'lambda_val', 1.0,
        'The gradients of the layer is multiplied by the negative lambda'
        'only during backpropagation. The lambda is used to scale'
        'gradients and not updated.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GradientReversalLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert isinstance(p.lambda_val, float)

  def FProp(self, inputs):
    """Gradients multiplied by a negative scalar during a backpropagation.

    Args:
      inputs: The inputs tensor.  Shaped [time, batch, input_dim].
    Returns:
      inputs
    """
    p = self.params
    with tf.name_scope(p.name):
      dtype = inputs.dtype

      @function.Defun(dtype, dtype)
      def FlipGradient(unused_x, dy):
        return (-p.lambda_val) * dy

      @function.Defun(dtype, grad_func=FlipGradient)
      def Identity(x):
        return x

      return Identity(inputs)


def RingBroadcast(input_tensor, target_devices):
  # Broadcast the input_tensor using a ring pattern to all target_devices.
  # The input_tensor is first sent to target_devices[0], which then forwards it
  # to target_devices[1], and so on upto target_devices[n-1].
  tensor_list = []
  prev_tensor = input_tensor
  for device in target_devices:
    with tf.device(device):
      tensor_list.append(tf.identity(prev_tensor))
    prev_tensor = tensor_list[-1]
  return tensor_list


def ShardedClassIdsToDense(class_ids, num_classes, num_shards, devices=None):
  """Convert class_ids into num_shards dense class labels."""
  assert num_classes % num_shards == 0
  if devices is None:
    devices = [''] * num_shards
  assert num_shards == len(devices)
  num_classes_per_shard = num_classes // num_shards

  class_ids = tf.reshape(class_ids, [-1])
  shard_class_ids = class_ids % num_classes_per_shard
  sharded_labels = []
  for i, dev in enumerate(devices):
    shard_min = i * num_classes_per_shard
    shard_max = (i + 1) * num_classes_per_shard
    in_shard = tf.cast(
        tf.logical_and(
            tf.less(class_ids, shard_max), tf.greater_equal(
                class_ids, shard_min)), tf.int32)
    indices = tf.cast(shard_class_ids * in_shard + (-1) * (1 - in_shard),
                      tf.int64)
    with tf.device(dev):
      sharded_labels.append(
          tf.one_hot(
              indices, depth=num_classes_per_shard, on_value=1.0,
              off_value=0.0))
  return sharded_labels


class SimpleFullSoftmax(SoftmaxLayer):
  """A somewhat simple softmax layer."""

  @classmethod
  def Params(cls):
    """Params for SimpleFullSoftmax."""
    p = super(SimpleFullSoftmax, cls).Params()
    p.Define(
        'num_sampled', 0, 'Number of samples to use for the sampled soft-max. '
        'Default value of 0 means no sampling is done; if set to > 0 then '
        'training will use sampled soft-max when both chunk_size == 0 and '
        'FProp is called with class_probabilities=None.')
    p.Define(
        'num_shards', 1,
        'Number of shards to split params into. num_shards should'
        ' divide num_classes.')
    p.Define('tie', False, 'Use tied embeddings from the embedding layer')
    # TODO(jmluo) HACKY
    p.Define('num_roles', 0, 'Number of roles. Used only by HRR')
    p.Define('gating', False, 'Flag to use gating for different roles')
    p.Define('anneal', False, 'Flag to reverse-anneal the temperature for softmax')
    p.Define('role_anneal', 0.0, 'Anneal the weight to 1.0 at this step')
    p.Define('dropout', False, 'Flag to use dropout on roles')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a SimpleFullSoftmax layer."""
    super(SimpleFullSoftmax, self).__init__(params)
    p = self.params
    assert p.name
    # We shard params across the class dimension.
    assert p.num_classes % p.num_shards == 0
    num_classes_per_shard = p.num_classes // p.num_shards
    # When using sampled soft-max we'd rather work with weights of
    # shape=[num_classes_per_shard, p.input_dim] to avoid an expensive transpose
    # op before computing the sampled_softmax_loss.
    self._transpose_weight_params = False
    self._weights_shard_shape = [p.input_dim, num_classes_per_shard]
    if p.num_sampled:
      self._transpose_weight_params = True
      self._weights_shard_shape = [num_classes_per_shard, p.input_dim]

    with tf.variable_scope(p.name):
      pc = py_utils.WeightParams(
          shape=self._weights_shard_shape,
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])

      if not p.tie:
        for i in range(p.num_shards):
          self.CreateVariable('weight_%d' % i, pc, self.AddGlobalVN)

      pc.shape = [num_classes_per_shard]
      pc.init.method = 'constant'
      pc.init.scale = 0.0
      for i in range(p.num_shards):
        if p.num_roles < 2:
          self.CreateVariable('bias_%d' % i, pc, self.AddGlobalVN)
        else:
          # This decomposes every bias term into num_roles terms -- useful for computing KL
          for role_ind in range(p.num_roles):
            self.CreateVariable('bias_r%d_%d' % (role_ind, i), pc,
                                self.AddGlobalVN)

      if p.gating:
        assert p.num_roles > 1
        pc = py_utils.WeightParams(
              shape=[p.input_dim // p.num_roles, p.num_roles],
              init=p.params_init,
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']) # TODO(jmluo) this is hacky
        self.CreateVariable('gating', pc, self.AddGlobalVN)#, trainable=False)

  def _CheckIfHasBias(self, theta):
    p = self.params
    if p.num_roles > 1:
      for i in range(p.num_shards):
        name = 'bias_%d' % i
        if name not in theta:
          all_biases = [
           0 * theta['bias_r%d_%d' % (role_ind, i)] # no biases
          # 0.5 * theta['bias_r%d_%d' % (role_ind, i)]
          for role_ind in range(p.num_roles)
          ]
          theta[name] = tf.reduce_sum(
                tf.stack(all_biases, axis=-1), axis=-1)


  def _GetInputs(self, inputs):
    if isinstance(inputs, list):
      assert len(inputs) == 1
      return inputs[0]
    return inputs

  def _ConcatWeights(self, theta):
    p = self.params
    # Add per-step noise if configured so.
    concat_axis = 1
    if self._transpose_weight_params:
      concat_axis = 0
    weights = [
        theta['weight_%d' % i]
        for i in range(p.num_shards)
    ]
    biases = [
        theta['bias_%d' % i]
        for i in range(p.num_shards)
    ]
    new_theta = theta.copy()
    new_theta.wm = py_utils.AddPerStepVN(p, tf.concat(
        weights, axis=concat_axis))
    new_theta.bias = py_utils.AddPerStepVN(p, tf.concat(biases, axis=0))
    return new_theta

  def _GetXWForRole(self, theta, inputs, role_ind):
    p = self.params
    preceding_shape = tf.shape(inputs)[:-1]
    inp = tf.reshape(inputs, tf.concat([preceding_shape, [p.num_roles, -1]], axis=0))[..., role_ind, :]
    last_dim = p.input_dim // p.num_roles
    if self._transpose_weight_params:
      w = tf.reshape(theta.wm, [-1, p.num_roles, last_dim])[:, role_ind]
    else:
      w = tf.reshape(theta.wm, [p.num_roles, last_dim, -1])[role_ind]
    res = py_utils.Matmul(inp, w, transpose_b=self._transpose_weight_params)
    return tf.reshape(res, tf.concat([preceding_shape, [-1]], axis=0))

  def _LogitsUsingConcatenatedWeights(self, theta, inputs, activation=None):
    p = self.params
    # inputs = self.ApplyClipping(theta, inputs)

    # x * w + b
    # Note that theta.wm and theta.bias are transformed to concated/clipped
    # by caller.
    if p.gating:
      assert activation is not None
      xws = list()
      for role_ind in xrange(p.num_roles):
        xws.append(self._GetXWForRole(theta, inputs, role_ind))
      gating_probs = self._GetGatingProbs(theta, activation)
      logits = tf.reduce_sum(tf.stack(xws, axis=-1) * tf.expand_dims(gating_probs, axis=-2), axis=-1)
      # logits = tf.reduce_sum(tf.stack(xws, axis=-1), axis=-1) / 2
      # logits = logits# + theta.bias
    else:
      logits =  tf.nn.bias_add(
        py_utils.Matmul(inputs, theta.wm, transpose_b=self._transpose_weight_params),
        theta.bias)

    # Clip logits by range.
    # Note that this is generally not used in conjunction with a clipping
    # schedule.
    abs_max = p.logits_abs_max
    if abs_max is not None:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = tf.clip_by_value(logits, abs_min, abs_max)

    # logits = self.ApplyClipping(theta, logits)
    if 'gating_probs' in locals() and p.is_eval:
      return logits, gating_probs
    else:
      return logits

  def _GetGatingProbs(self, theta, activation):
    last_dim = tf.shape(activation)[-1]
    inp = tf.reshape(activation, [-1, last_dim])
    logits = py_utils.Matmul(inp, theta.gating)
    # probs = tf.sigmoid(logits)
    p = self.params
    if p.anneal:
      global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
      temperature = 10.0 - 9.9 * (tf.minimum(tf.constant(3000.0), global_step) / 3000)
      # temperature = 10.0
      probs = tf.nn.softmax(logits / temperature)
    if p.role_anneal > 0:
      preceding_shape = tf.shape(inp)[:-1]
      prob_1 = tf.ones(shape=preceding_shape)
      global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
      # ra = tf.to_float(p.role_anneal)
      # slope = 1.0 / (ra - 1000.0)
      # temperature = tf.cond(global_step > 1000.0, lambda: (ra - 1000.0) * slope, lambda: 0.0)
      # temperature = tf.minimum(1.0, temperature)
      temperature = tf.minimum(tf.constant(p.role_anneal * 1.0), global_step) / (p.role_anneal * 1.0)
      tf.summary.scalar('temperature', temperature)
      probs = tf.stack([prob_1, prob_1 * temperature], axis=-1)
    if p.dropout:
      preceding_shape = tf.shape(inp)[:-1]
      rn = tf.random_uniform(shape=tf.concat([preceding_shape, [1]], axis=0))
      case1 = tf.to_float(rn < 0.25)
      case3 = tf.to_float(rn > 0.5)
      case2 = 1.0 - case1 - case3
      prob_0 = tf.zeros(shape=preceding_shape)
      prob_1 = tf.ones(shape=preceding_shape)

      weight1 = tf.stack([prob_1 * 2.0, prob_0], axis=-1)
      weight2 = tf.stack([prob_0, prob_1 * 2.0], axis=-1)
      weight3 = tf.stack([prob_1, prob_1], axis=-1)
      if p.is_eval:
        probs = weight3
      else:
        probs = case1 * weight1 + case2 * weight2 + case3 * weight3
    return probs

  def Logits(self, theta, inputs, activation=None, return_gating=False):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape
        [N, input_dim].

    Returns:
      logits [batch, num_classes]
    """
    self._CheckIfHasBias(theta)
    logits = self._LogitsUsingConcatenatedWeights(
        self._ConcatWeights(theta), self._GetInputs(inputs), activation=activation)
    if isinstance(logits, tuple):
      logits, gating_probs = logits
      if return_gating:
        return logits, gating_probs
    return logits

  def _XentLossByChunk(self, theta, activation, class_ids):
    """Computes per-example xent loss between activation and class_ids."""
    p = self.params

    # We reshape activation from a matrix to a 3-D tensor (a sequence
    # of matrices), where the 2nd dimenion is p.chunk_size.  Because
    # the batch dimenion may not be multiple of p.chunk_size, we pad
    # zeros.
    activation = py_utils.HasRank(activation, 2)
    batch, input_dim = tf.unstack(tf.shape(activation))
    dim0, dim1 = (batch + p.chunk_size - 1) // p.chunk_size, p.chunk_size
    pad = dim0 * dim1 - batch
    padded_activation = tf.concat(
        [activation,
         tf.zeros([pad, input_dim], dtype=activation.dtype)],
        axis=0)
    class_ids = py_utils.HasShape(class_ids, [batch, 1])
    padded_class_ids = tf.concat(
        [class_ids, tf.zeros([pad, 1], dtype=class_ids.dtype)], axis=0)

    if py_utils.use_tpu():
      id_dtype = tf.int32
    else:
      id_dtype = tf.int64
    padded_class_ids = tf.cast(padded_class_ids, id_dtype)

    # For each chunk, we compute logits of padded_activation[i, :, :],
    # and its xent loss with padded_class_ids[i, :].
    def ChunkFn(theta, state0, inputs):
      del state0
      activation, class_ids = inputs.activation, inputs.class_ids
      logits = self._LogitsUsingConcatenatedWeights(theta, activation)
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=class_ids)
      amax = tf.stop_gradient(py_utils.ArgMax(logits))
      return py_utils.NestedMap(xent=xent, amax=amax), py_utils.NestedMap()

    acc, _ = recurrent.Recurrent(
        theta=self._ConcatWeights(theta),
        state0=py_utils.NestedMap(
            xent=tf.zeros([p.chunk_size], dtype=p.dtype),
            amax=tf.zeros([p.chunk_size], dtype=id_dtype)),
        inputs=py_utils.NestedMap(
            activation=tf.reshape(padded_activation, [dim0, dim1, input_dim]),
            class_ids=tf.reshape(padded_class_ids, [dim0, dim1])),
        cell_fn=ChunkFn)

    # acc.xent has the shape [dim0, dim1]. acc.xent[i, :] are
    # per-example xent loss for examples in the i-th chunk.  We
    # reshape acc.xent to a vector and slice the first 'batch' values.
    def GetBatch(x):
      return tf.reshape(x, [-1])[:batch]

    return GetBatch(acc.xent), GetBatch(acc.amax)

  def _FProp2D(self,
               theta,
               inputs,
               class_weights,
               class_ids=None,
               class_probabilities=None,
               activation=None):
    """Computes xent loss and log-prob logit."""
    p = self.params
    inputs = self._GetInputs(inputs)
    logits = self.Logits(theta, inputs, activation=activation)

    if class_probabilities is not None:
      per_example_xent = tf.nn.softmax_cross_entropy_with_logits(
          labels=class_probabilities, logits=logits)
      per_example_argmax = py_utils.ArgMax(logits)
    elif p.chunk_size:
      class_ids = py_utils.HasShape(class_ids, [-1, 1])
      per_example_xent, per_example_argmax = self._XentLossByChunk(
          theta, inputs, class_ids)
    elif p.num_sampled is 0 or p.is_eval:
      assert class_ids is not None
      assert logits is not None
      tf.logging.vlog(
          0, 'Using sparse_softmax_cross_entropy_with_logits() in '
          'SimpleFullSoftmax::_FProp2D logits_shape=%r',
          py_utils.GetShape(logits))
      per_example_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(class_ids, [-1]), logits=logits)
      per_example_argmax = py_utils.ArgMax(logits)
    else:  # Use sampled soft-max in training mode with p.num_sampled set.
      assert p.num_sampled > 0
      tf.logging.vlog(
          0, 'Using sampled_softmax_loss(..., num_sampled=%d, '
          'num_classes=%d) in SimpleFullSoftmax::_FProp2D', p.num_sampled,
          p.num_classes)

      if p.num_roles < 2:
        per_example_xent = tf.nn.sampled_softmax_loss(
            weights=[theta['weight_%d' % i] for i in range(p.num_shards)],
            biases=tf.concat(
                [theta['bias_%d' % i] for i in range(p.num_shards)], axis=0),
            labels=tf.reshape(class_ids, [-1, 1]),
            inputs=self._GetInputs(inputs),
            num_sampled=p.num_sampled,
            num_classes=p.num_classes,
            partition_strategy='div')
      else:
        num_classes_per_shard = theta.weight_0.get_shape().as_list()[0]
        all_weights = [
            tf.reshape(theta['weight_%d' % i], [num_classes_per_shard, p.num_roles, -1])
            for i in range(p.num_shards)
        ]
        all_inputs = self._GetInputs(inputs)
        preceding_shape = tf.shape(all_inputs)[:-1]
        all_inputs = tf.reshape(all_inputs, tf.concat([preceding_shape, [p.num_roles, -1]], axis=0))
        reshaped_labels = tf.reshape(class_ids, [-1, 1])
        if reshaped_labels.dtype != tf.int64:
          reshaped_labels = tf.cast(reshaped_labels, tf.int64)
        sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes=reshaped_labels,
            num_true=1,
            num_sampled=p.num_sampled,
            unique=True,
            range_max=p.num_classes)
        all_logits = list()
        for role_ind in range(p.num_roles):
          w_i = [w[:, role_ind] for w in all_weights]
          b_i = tf.concat(
              [
                  0 * theta['bias_r%d_%d' % (role_ind, i)] # no biases
                  # 0.5 * theta['bias_r%d_%d' % (role_ind, i)]
                  for i in range(p.num_shards)
              ],
              axis=0)
          inp_i = all_inputs[..., role_ind, :]
          subtract = role_ind == 0
          r_logits, labels = sampled_softmax_logits(
              weights=w_i,
              biases=b_i,
              labels=reshaped_labels,
              sampled_values=sampled_values,
              inputs=inp_i,
              num_sampled=p.num_sampled,
              num_classes=p.num_classes,
              subtract_log_q=subtract,
              partition_strategy='div')
          all_logits.append(r_logits)

        if p.gating:
          assert activation is not None
          gating_probs = self._GetGatingProbs(theta, activation)
          summed_logits = tf.reduce_sum(tf.stack(all_logits, axis=-1) * tf.expand_dims(gating_probs, axis=-2), axis=-1)
          # summed_logits = tf.reduce_sum(tf.stack(all_logits, axis=-1), axis=-1) / 2
        else:
          # summed_logits = sum(all_logits)
          summed_logits = tf.reduce_sum(tf.stack(all_logits, axis=-1), axis=-1)
        per_example_xent = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=summed_logits)

      # Avoid computing logits; per_example_argmax is going to be always right.
      per_example_argmax = tf.identity(class_ids)


    def _FPropDtype(params):
      return params.fprop_dtype if params.fprop_dtype is not None else params.dtype

    label_weights = tf.reshape(tf.cast(class_weights, _FPropDtype(p)), [-1])
    total_xent = tf.reduce_sum(per_example_xent * label_weights)
    total_weights = tf.reduce_sum(label_weights)

    all_logits = None if 'all_logits' not in locals() else all_logits
    return py_utils.NestedMap(
        logits=logits,
        log_probs=tf.nn.log_softmax(logits),
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent,
        per_example_weight=label_weights,
        total_xent=total_xent,
        total_weight=total_weights,
        avg_xent=total_xent / total_weights,
        all_logits=all_logits)


class MixtureOfSimpleSoftmax(SimpleFullSoftmax):
  """A mixture of simple softmax layers."""

  @classmethod
  def Params(cls):
    """Params for ShardedFullSoftmax."""
    p = super(MixtureOfSimpleSoftmax, cls).Params()
    p.Define('num_softmax', 1, 'Number of softmaxes to mix.')
    p.name = 'mos_softmax'
    return p

  @base_layer.initializer
  def __init__(self, params):  # pylint: disable=super-init-not-called
    """Constructs a DotProductSoftmax layer."""
    # Skipping SimpleFullSoftmax constructor as it creates unnecessary
    # variables.
    SoftmaxLayer.__init__(self, params)  # pylint: disable=non-parent-init-called
    p = params
    assert p.name
    input_softmax_p = SimpleFullSoftmax.Params()
    input_softmax_p.input_dim = p.input_dim
    input_softmax_p.num_classes = p.num_classes
    input_softmax_p.num_shards = p.num_shards

    softmax_params = []
    for i in range(p.num_softmax):
      input_softmax_p_i = input_softmax_p.Copy()
      input_softmax_p_i.name = 'input_softmax_%d' % i
      softmax_params += [input_softmax_p_i]

    self.CreateChildren('softmax', softmax_params)

    gate_softmax_p = SimpleFullSoftmax.Params()
    gate_softmax_p.name = 'gate'
    gate_softmax_p.input_dim = p.input_dim
    gate_softmax_p.num_classes = p.num_softmax

    self.CreateChild('gate_softmax', gate_softmax_p)

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape
        [N, input_dim].
    """
    p = self.params
    mixture_logits = list([
        softmax.Logits(softmax_theta, inputs)
        for (softmax, softmax_theta) in zip(self.softmax, theta.softmax)
    ])
    probs = tf.stack(
        list([tf.nn.softmax(logits) for logits in mixture_logits]),
        -2,
        name='mixture_probs')
    gate_probs = tf.nn.softmax(
        self.gate_softmax.Logits(theta.gate_softmax, inputs), name='gate_probs')
    combined_probs = tf.squeeze(
        tf.matmul(tf.expand_dims(gate_probs, -2), probs, name='combined_probs'),
        axis=-2)
    combined_logits = tf.log(combined_probs, name='combined_logits')

    # For Tensorboard Summary
    avg_gate_probs = tf.reduce_mean(
        tf.reshape(gate_probs, [-1, p.num_softmax]), axis=0)
    summary_utils.scalar(p, 'softmax_most_popular_expert',
                         tf.reduce_max(avg_gate_probs))
    summary_utils.scalar(p, 'softmax_least_popular_expert',
                         tf.reduce_min(avg_gate_probs))
    summary_utils.histogram(p, 'softmax_gate_popularity', avg_gate_probs)
    return combined_logits


class MixtureOfSoftmaxShareProjection(SimpleFullSoftmax):
  """A mixture of softmax layers (https://arxiv.org/abs/1711.03953).

     Each expert owns a hidden layer that is applied before the final projection
     layer. The final projection layer is shared between experts.
  """

  @classmethod
  def Params(cls):
    """Params for MixtureOfSoftmaxShareProjection."""
    p = super(MixtureOfSoftmaxShareProjection, cls).Params()
    p.Define('num_softmax', 1, 'Number of softmax experts to mix.')
    p.name = 'mos_softmax'
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a MixtureOfSoftmaxShareProjection layer."""
    super(MixtureOfSoftmaxShareProjection, self).__init__(params)
    p = params
    assert p.name

    expert_params = []
    expert_p = ProjectionLayer.Params()
    expert_p.input_dim = p.input_dim
    expert_p.output_dim = p.input_dim
    expert_p.activation = 'TANH'
    expert_p.batch_norm = False
    expert_p.params_init = p.params_init
    for i in range(p.num_softmax):
      expert_p_i = expert_p.Copy()
      expert_p_i.name = 'softmax_expert_%d' % i
      expert_params += [expert_p_i]

    self.CreateChildren('experts', expert_params)

    gate_softmax_p = SimpleFullSoftmax.Params()
    gate_softmax_p.name = 'gate'
    gate_softmax_p.input_dim = p.input_dim
    gate_softmax_p.num_classes = p.num_softmax
    gate_softmax_p.params_init = p.params_init

    self.CreateChild('gate_softmax', gate_softmax_p)

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of a single tensor, or a single tensor with the shape
        [N, input_dim].
    """
    p = self.params
    inputs = self._GetInputs(inputs)
    expert_outputs = list([
        expert.FProp(expert_theta, inputs)
        for (expert, expert_theta) in zip(self.experts, theta.experts)
    ])
    mixture_logits = list([
        super(MixtureOfSoftmaxShareProjection, self).Logits(
            theta, expert_output) for expert_output in expert_outputs
    ])
    probs = tf.stack(
        list([tf.nn.softmax(logits) for logits in mixture_logits]),
        -2,
        name='mixture_probs')
    gate_probs = tf.nn.softmax(
        self.gate_softmax.Logits(theta.gate_softmax, inputs), name='gate_probs')
    combined_probs = tf.squeeze(
        tf.matmul(tf.expand_dims(gate_probs, -2), probs, name='combined_probs'),
        axis=-2)
    combined_logits = tf.log(combined_probs, name='combined_logits')

    # For Tensorboard Summary
    avg_gate_probs = tf.reduce_mean(
        tf.reshape(gate_probs, [-1, p.num_softmax]), axis=0)
    summary_utils.scalar(p, 'softmax_most_popular_expert',
                         tf.reduce_max(avg_gate_probs))
    summary_utils.scalar(p, 'softmax_least_popular_expert',
                         tf.reduce_min(avg_gate_probs))
    summary_utils.histogram(p, 'softmax_gate_popularity', avg_gate_probs)
    return combined_logits


class SpuSimpleFullSoftmax(SimpleFullSoftmax):
  """A SPU version of simple softmax layer."""

  @classmethod
  def Params(cls):
    """Params for SpuSimpleFullSoftmax."""
    p = super(SpuSimpleFullSoftmax, cls).Params()
    p.Define('input_ranges', (-1.0, 1.0), 'The ranges for the input.')
    p.Define('input_quant_dtype', tf.quint16, 'The quantized dtype for input.')
    p.Define('weight_quant_dtype', tf.qint8, 'The quantized dtype for weight.')
    p.Define('bias_quant_dtype', tf.qint32, 'The quantized dtype for bias.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Init the SpuSimpleFullSoftmax layer."""
    super(SpuSimpleFullSoftmax, self).__init__(params)
    p = self.params
    assert p.name

    with tf.variable_scope(p.name):
      w_shape = [p.input_dim, p.num_classes]
      b_shape = [p.num_classes]
      wm = tf.concat([self.vars['weight_%d' % i] for i in range(p.num_shards)],
                     1)
      bias = tf.concat([self.vars['bias_%d' % i] for i in range(p.num_shards)],
                       0)
      with tf.device(py_utils.SpuDevice()):
        self._quantized_w = py_utils.CreateQuantVariable(
            '%s_quantized_w' % p.name, wm, p.weight_quant_dtype, w_shape)
        self._quantized_b = py_utils.CreateQuantBias(
            '%s_quantized_b' % p.name, bias, p.bias_quant_dtype, b_shape)

  @property
  def spu_init_op(self):
    """Return the spu_init_op for softmax."""
    return tf.group(*[self._quantized_w.init, self._quantized_b.init])

  def Logits(self, unused_theta, inputs, inputs_shape=None):
    """Returns the logits and log_probs.

    Args:
      unused_theta: A nested map object containing weights' values of this
        layer and its children layers, but here not being used.
      inputs: a list of a single tensor, or a single tensor with the shape
        [N, input_dim].
      inputs_shape: optional, shape of the inputs but aquired by a CPU node.

    Returns:
      logits and log_probs [batch, num_classes]

    Raises:
      ValueError: 'inputs' contains more than one tensor or 'inputs'
      dtype not match specified input_quant_dtype.
    """
    p = self.params
    if isinstance(inputs, list):
      if len(inputs) != 1:
        raise ValueError('inputs list expects to have only one tensor.')
      inputs = inputs[0]

    if inputs.dtype != p.input_quant_dtype:
      raise ValueError('inputs dtype %s does not match input_quant_dype %s.' %
                       (inputs.dtype, p.input_quant_dtype))

    if inputs_shape is None:
      inputs_shape = tf.shape(inputs)
    ones = tf.ones(shape=inputs_shape, dtype=tf.float32)
    ones_q, _, _ = array_ops.quantize(ones, 0.0, 1.0, tf.quint16)

    input_min = p.input_ranges[0]
    input_max = p.input_ranges[1]
    with tf.name_scope(p.name):
      with tf.device(py_utils.SpuDevice()):
        # Convert 16-bit to 8-bit for 8-bit Matmul.
        if (p.input_quant_dtype == tf.quint16 and
            p.weight_quant_dtype == tf.qint8):
          inputs = internal_math_ops.mul_n_clip(
              [inputs, ones_q], [input_min, 0.0], [input_max, 1.0],
              out_type=tf.quint8,
              output_min=input_min,
              output_max=input_max)
        xw, _, _ = internal_math_ops.quantized_mat_mul_v2(
            a=inputs,
            b=self._quantized_w.var,
            min_a=input_min,
            max_a=input_max,
            weight_scale=self._quantized_w.scale,
            Toutput=tf.qint32)
        unused = 1.0
        output, _, _ = nn_ops.quantized_bias_add(
            input=xw,
            bias=self._quantized_b.var,
            min_input=-unused,
            max_input=unused,
            min_bias=-unused,
            max_bias=unused,
            out_type=tf.qint32)
        outputs, _, _ = internal_math_ops.quantized_hard_tanh_x(
            output, max_value=p.logits_abs_max, min_features=-1, max_features=1)
      logits = tf.dequantize(outputs, -p.logits_abs_max, p.logits_abs_max)
      log_probs = nmt_py_x_ops.logp_from_quantized_logits(
          outputs, -p.logits_abs_max, p.logits_abs_max)
    return logits, log_probs


class ShardedFullSoftmax(SoftmaxLayer):
  """A softmax that is sharded to multiple gpus."""

  @classmethod
  def Params(cls):
    """Params for ShardedFullSoftmax."""
    p = super(ShardedFullSoftmax, cls).Params()
    p.Define(
        'num_shards', 0,
        'Number of shards to split params into. num_shards should'
        ' divide num_classes.')
    p.Define(
        'local_devices', [],
        'A list of worker devices to compute the softmax on.'
        ' The number of worker devices should divide num_shards.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Constructs a DotProductSoftmax layer."""
    super(ShardedFullSoftmax, self).__init__(params)
    p = self.params
    assert p.name
    assert p.num_classes % p.num_shards == 0
    num_classes_per_shard = p.num_classes // p.num_shards

    with tf.variable_scope(p.name):
      # We shard params across columns.
      pc = py_utils.WeightParams(
          shape=[p.input_dim, num_classes_per_shard],
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
      for i in range(p.num_shards):
        self.CreateVariable('weight_%d' % i, pc, self.AddGlobalVN)

      pc.shape = [num_classes_per_shard]
      pc.init.method = 'constant'
      pc.init.scale = 0.0
      for i in range(p.num_shards):
        self.CreateVariable('bias_%d' % i, pc, self.AddGlobalVN)

  def _LocalWeightsAndBiases(self, theta):
    p = self.params
    assert p.num_shards % len(p.local_devices) == 0
    num_shards_per_local_dev = p.num_shards // len(p.local_devices)

    weights, biases = [], []
    for dev_id, dev in enumerate(p.local_devices):
      with tf.device(dev):
        first_shard = dev_id * num_shards_per_local_dev
        last_shard = (dev_id + 1) * num_shards_per_local_dev
        # TODO(yonghui): Get rid of this concat.
        weights.append(
            tf.concat([
                theta['weight_%d' % i] for i in range(first_shard, last_shard)
            ], 1))
        biases.append(
            tf.concat(
                [theta['bias_%d' % i] for i in range(first_shard, last_shard)],
                0))
    return weights, biases

  def _LogitsShard(self, inputs, weight, bias):
    p = self.params
    # Add per-step noise if configured so.
    weight = py_utils.AddPerStepVN(p, weight)
    bias = py_utils.AddPerStepVN(p, bias)
    logits = tf.nn.bias_add(py_utils.Matmul(inputs, weight), bias)
    abs_max = p.logits_abs_max
    if abs_max is not None:
      abs_min = -abs_max  # pylint: disable=invalid-unary-operand-type
      logits = tf.clip_by_value(logits, abs_min, abs_max)
    return logits

  def Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of tensors of length equal to number of words in the
          sentence (seq_len), each with shape [batch_size, input_dim].

    Returns:
      logits [batch, num_classes]
    """
    sharded_logits = self._Logits(theta, inputs)[0]
    unnormalized_logits = tf.concat(sharded_logits, axis=1)
    return unnormalized_logits

  def _Logits(self, theta, inputs):
    """Returns the logits computed before the softmax.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a list of tensors of length equal to number of words in the
          sentence (seq_len), each with shape [batch_size, input_dim].

    Returns:
      tuple of (
          List of logits for each softmax shard.
          List of max values from the logits in each softmax shard.
      )
    """
    p = self.params

    def BroadcastInputs(inputs):
      """Broadcast each tensor in the inputs list to all devices on worker."""
      # An efficient way to do the broadcast is to forward the tensors using a
      # ring communication pattern to all the devices, each node on the ring
      # being a device. Since the output of the decoder is produced on the last
      # device, the communication pattern would be as follows:
      # Node(n-1) --> Node(n-2) --> ... --> Node(1) --> Node(0).
      target_devices = list(p.local_devices)
      target_devices.reverse()
      # shard_local_inputs is a 2D list. The outer dimension corresponds to the
      # different decoder steps. The inner dimension corresponds to the
      # different local worker nodes (CPUs/GPUs).
      shard_local_inputs = []
      for input_tensor in inputs:
        broadcast_inputs = RingBroadcast(input_tensor, target_devices)
        broadcast_inputs.reverse()
        shard_local_inputs.append(broadcast_inputs)
      return shard_local_inputs

    shard_local_inputs = BroadcastInputs(inputs)

    # Concatenate the inputs corresponding to the same worker node.
    # To do this, the 2D list is transposed first such that the outer dimension
    # corresponds to the different local worker nodes (CPUs/GPUs), and the inner
    # dimension to the various decoder steps.
    # Note that the broadcast is performed first, followed by the concat to
    # overlap the communication with the decoder computation.
    shard_local_fused_inputs = []
    if len(inputs) > 1:
      for dev_id, dev in enumerate(p.local_devices):
        dev_inputs = []
        for step_id in range(len(inputs)):
          dev_inputs.append(shard_local_inputs[step_id][dev_id])
        with tf.device(dev):
          shard_local_fused_inputs.append(
              tf.reshape(tf.concat(dev_inputs, 1), [-1, p.input_dim]))
    else:
      assert len(shard_local_inputs) == 1
      shard_local_fused_inputs = shard_local_inputs[0]

    weights, biases = self._LocalWeightsAndBiases(theta)

    inputs = shard_local_fused_inputs
    compute_devices = p.local_devices
    assert len(inputs) == len(weights) == len(biases) == len(compute_devices)

    with tf.name_scope(p.name):
      sharded_logits = []
      sharded_logits_max = []

      # We shard across columns.
      for shard_id, dev in enumerate(compute_devices):
        with tf.name_scope('shard_%d' % (shard_id)):
          with tf.device(dev):
            w = weights[shard_id]
            b = biases[shard_id]
            tf.logging.vlog(1, 'Placing softmax computation on device: %s', dev)
            logits = self._LogitsShard(inputs[shard_id], w, b)
            sharded_logits.append(logits)
            sharded_logits_max.append(tf.reduce_max(logits, 1, keepdims=True))

    return sharded_logits, sharded_logits_max

  def _XentLossInternal(self,
                        sharded_logits,
                        sharded_logits_max,
                        class_weights,
                        class_ids=None,
                        class_probabilities=None):
    """Computes xent loss and log-prob logit.

    This function computes the cross entropy loss across multiple devices.
    The inputs (/activations) are broadcasted to all the compute_devices.
    The weights and biases are sharded across the compute_devices. Note that
    the inputs, weights, biases and compute_devices are lists of the same size.
    Each element of the inputs list is the same replicated activation tensor
    that is local to the corresponding compute_device. Each element of the
    weights and biases list corresponds to the parameter shards that are placed
    on that device.

    Args:
      sharded_logits: List of logits for each softmax shard.
      sharded_logits_max: List of max values from the logits in each shard.
      class_weights: a tensor with shape [batch_size * seq_len] of float dtype
          that contains the weights for each target word.
      class_ids: a tensor with shape [batch_size * seq_len] of int32 dtype that
          contains the target class labels. Must be None when
          class_probabilities are provided.
      class_probabilities: a tensor with shape
          [batch_size * seq_len * num_classes] of float32 dtype that contains
          the distribution over target class labels. Must be None if class_ids
          are provided.

    Returns:
      A dict of final and intermediate tensors.
    """
    p = self.params
    dtype = sharded_logits[0].dtype
    num_classes = p.num_classes
    compute_devices = p.local_devices
    assert (class_ids is None) + (class_probabilities is None) == 1, (
        'please provide exactly one of class_ids or class_probabilities')

    with tf.name_scope(p.name):
      logits_max = tf.stop_gradient(
          tf.reduce_max(tf.concat(sharded_logits_max, 1), 1, keepdims=True))

      sharded_logits_sum_exp = []
      for i, dev in enumerate(compute_devices):
        with tf.device(dev):
          sharded_logits_sum_exp.append(
              tf.reduce_sum(
                  tf.exp(sharded_logits[i] - logits_max), 1, keepdims=True))

      logits_sum_exp = tf.stop_gradient(tf.add_n(sharded_logits_sum_exp))

      if class_ids is not None:
        all_in_range = py_utils.Assert(
            tf.reduce_all(
                tf.logical_and(
                    tf.less(class_ids, num_classes),
                    tf.greater_equal(class_ids, 0))), [class_ids])
        class_ids = py_utils.with_dependencies([all_in_range], class_ids)
        sharded_labels = ShardedClassIdsToDense(class_ids, num_classes,
                                                len(compute_devices),
                                                compute_devices)
      else:
        class_probabilities = py_utils.with_dependencies([
            py_utils.assert_equal(
                tf.shape(class_probabilities)[-1], p.num_classes)
        ], class_probabilities)
        flat_labels = tf.reshape(class_probabilities,
                                 [-1, tf.shape(class_probabilities)[-1]])
        sharded_labels = tf.split(
            flat_labels, num_or_size_splits=len(compute_devices), axis=1)

      sharded_xent = []
      for i, dev in enumerate(compute_devices):
        labels_i = sharded_labels[i]
        if dtype != tf.float32:
          labels_i = tf.cast(labels_i, dtype)
        with tf.device(dev):
          xent_i = nmt_py_x_ops.softmax_cross_entropy_with_logits_shard(
              sharded_logits[i], tf.reshape(logits_max, [-1]),
              tf.reshape(logits_sum_exp, [-1]), labels_i)
          sharded_xent.append(tf.reshape(xent_i, [-1, 1]))
      per_example_xent_2d = tf.add_n(sharded_xent)
      per_example_xent = tf.reshape(per_example_xent_2d, [-1])
      label_weights = tf.reshape(tf.cast(class_weights, dtype), [-1])
      total_xent = tf.reduce_sum(per_example_xent * label_weights)
      total_weights = tf.reduce_sum(label_weights)
      unnormalized_logits = tf.concat(sharded_logits, 1)

    return py_utils.NestedMap(
        logits=unnormalized_logits,
        log_probs=tf.nn.log_softmax(unnormalized_logits),
        per_example_argmax=py_utils.ArgMax(unnormalized_logits),
        per_example_xent=per_example_xent,
        per_example_weight=label_weights,
        total_xent=total_xent,
        total_weight=total_weights,
        avg_xent=total_xent / (total_weights + 1e-6))  # avoid dividing by 0.

  def _FProp2D(self,
               theta,
               inputs,
               class_weights,
               class_ids=None,
               class_probabilities=None):
    """Computes xent loss and log-prob logit on the local worker devices."""
    assert (class_ids is None) + (class_probabilities is None) == 1, (
        'please provide exactly one of class_ids or class_probabilities')

    sharded_logits, sharded_logits_max = self._Logits(theta, inputs)

    return self._XentLossInternal(
        sharded_logits,
        sharded_logits_max,
        class_weights,
        class_ids=class_ids,
        class_probabilities=class_probabilities)


class FirstNLayer(base_layer.LayerBase):
  """Returns the first n args."""

  @classmethod
  def Params(cls):
    p = super(FirstNLayer, cls).Params()
    p.Define('n', 0, 'The number of args to return.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(FirstNLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.n > 0

  def FProp(self, theta, *args):
    """Return the first n args."""
    p = self.params
    assert len(args) >= p.n
    return tuple(args[:p.n]) if p.n > 1 else args[0]


class GroupNormLayer(base_layer.LayerBase):
  """Group normalization layer(https://arxiv.org/abs/1803.08494)."""

  @classmethod
  def Params(cls):
    p = super(GroupNormLayer, cls).Params()
    p.Define('dim', 0, 'Depth of the input/output.')
    p.Define('num_groups', 32, 'Number of groups for GroupNorm.')
    p.Define('min_group_size', 1, 'Minimum group size for GroupNorm')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GroupNormLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.num_groups > 0
    assert p.min_group_size > 0
    assert p.dim >= p.num_groups, 'p.dim({0}) >= p.num_groups({1})'.format(
        p.dim, p.num_groups)
    assert p.dim % p.num_groups == 0, ('p.dim({0}) is not dividable by '
                                       'p.num_groups({1})').format(
                                           p.dim, p.num_groups)

    pc = py_utils.WeightParams(
        shape=[1, 1, 1, p.dim],
        init=py_utils.WeightInit.Constant(0.0),
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      self.CreateVariable('beta', pc)
      # Note, The real gamma to use is 1 + gamma.
      self.CreateVariable('gamma', pc, lambda x: 1.0 + x)

    self._epsilon = 0.001

  def FProp(self, theta, inputs):
    """Apply group normalization.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor with shape [batch_size, height, width, channel].
    Returns:
      Output after applying group normalization, with the same shape as
      'inputs'.
    """
    p = self.params
    n, h, w, c = tf.unstack(tf.shape(inputs), axis=0, num=4)
    group_size = p.dim // p.num_groups
    num_groups = p.num_groups
    min_group_size = p.min_group_size if p.dim > p.min_group_size else p.dim
    if group_size <= min_group_size:
      group_size = min_group_size
      num_groups = p.dim // group_size

    with tf.name_scope(p.name):
      x = tf.reshape(inputs, [n, h, w, num_groups, group_size])
      counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
          x, axes=[1, 2, 4], keep_dims=True)
      norm_mean, norm_variance = tf.nn.normalize_moments(
          counts, means_ss, variance_ss, None)

      norm_mean = py_utils.CheckNumerics(
          norm_mean, 'mean of %s failed numeric check' % p.name)
      norm_variance = py_utils.CheckNumerics(
          norm_variance, 'variance of %s failed numeric check' % p.name)

      beta = theta.beta
      gamma = theta.gamma

      with tf.control_dependencies([
          py_utils.assert_greater_equal(norm_variance, tf.cast(0., p.dtype)),
          assert_shape_match([n, 1, 1, num_groups, 1], tf.shape(norm_mean)),
          assert_shape_match([n, 1, 1, num_groups, 1], tf.shape(norm_variance)),
      ]):
        x = (x - norm_mean) / tf.sqrt(norm_variance + self._epsilon)
        x = tf.reshape(x, [n, h, w, c])
        gn_output = x * gamma + beta
        gn_output = tf.reshape(gn_output, [n, h, w, c])
        return gn_output

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    flops_per_element = 10  # Approximately 10 flops per element.
    return py_utils.NestedMap(
        flops=inputs.num_elements() * flops_per_element, out_shapes=(inputs,))


class ConvTransposeLayer(base_layer.LayerBase):
  """Convolution transpose layer."""

  @classmethod
  def Params(cls):
    p = super(ConvTransposeLayer, cls).Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), out_channel,'
        ' in_channel.')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the time dimension. The second int'
        ' specifies the stride on the frequency dimension.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ConvTransposeLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[p.filter_shape[-2]],
              init=py_utils.WeightInit.Constant(0.0),
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  def OutShape(self, in_shape):
    """Compute the output shape given the input shape."""
    if isinstance(in_shape, tf.TensorShape):
      return tf.TensorShape(self._OutShapeDimensions(in_shape))
    else:
      return tf.stack(self._OutShapeDimensions(in_shape))

  def _OutShapeDimensions(self, in_shape):
    """Compute the output shape dimensions given the input shape."""
    p = self.params
    t_stride = p.filter_stride[0]
    f_stride = p.filter_stride[1]
    return [
        in_shape[0], in_shape[1] * t_stride, in_shape[2] * f_stride,
        p.filter_shape[2]
    ]

  def _ApplyConv(self, theta, inputs):
    p = self.params
    w = theta.w
    strides = [1, p.filter_stride[0], p.filter_stride[1], 1]
    # TODO(miachen): remove casting once tf.nn.conv2d supports tf.float64.
    assert inputs.dtype == w.dtype
    dtype = inputs.dtype
    if dtype != tf.float32:
      inputs = tf.cast(inputs, tf.float32)
      w = tf.cast(w, tf.float32)
    out = tf.nn.conv2d_transpose(
        inputs,
        w,
        output_shape=self.OutShape(tf.shape(inputs)),
        strides=strides,
        padding='SAME')
    if dtype != tf.float32:
      out = tf.cast(out, dtype)
    out = tf.nn.bias_add(out, theta.b)
    return py_utils.HasShape(out, [-1, -1, -1, p.filter_shape[2]])

  def FProp(self, theta, inputs, paddings):
    """Apply convolution to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
          frequency, channel]. The time dimension corresponds to the height
          dimension as in images and the frequency dimension corresponds to the
          width dimension as in images.
      paddings: The paddings tensor. It is expected to be of shape [batch,
          time].
    Returns:
      outputs, out_paddings pair. outputs is expected to have shape [batch,
      time * time_stride, frequency * freq_stride, out_channel].
    """
    p = self.params
    inputs = py_utils.with_dependencies([
        assert_shape_match(tf.shape(paddings), [-1, -1]),
        assert_shape_match(
            tf.shape(inputs),
            tf.concat([tf.shape(paddings), [-1, p.filter_shape[3]]], 0))
    ], inputs)
    with tf.name_scope(p.name):
      out = self._ApplyConv(theta, inputs)
      # [B, T, 1].
      out_padding = tf.expand_dims(paddings, -1)
      # [B, T, stride].
      out_padding = tf.tile(out_padding, [1, 1, p.filter_stride[0]])
      # [B, T * stride].
      out_padding = tf.reshape(out_padding, [tf.shape(paddings)[0], -1])
      # Lastly zeroing out padded states.
      out *= tf.expand_dims(tf.expand_dims(1.0 - out_padding, -1), -1)
      return out, out_padding


class GatedConvLayer(base_layer.LayerBase):
  """GatedConvolution layer, with optional batch normalization.

  Gated convolution layer as used in the wavenet paper:
  https://arxiv.org/pdf/1609.03499.pdf
  """

  @classmethod
  def Params(cls):
    p = super(GatedConvLayer, cls).Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' out_channel.')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the time dimension. The second int'
        ' specifies the stride on the frequency dimension.')
    p.Define('activation', 'TANH',
             'Activation function to use. Options are TANH, NONE.')
    p.Define('batch_norm', True, 'Whether or not to apply batch norm.')
    p.Define(
        'bn_decay', 0.999,
        'Decay in updating the mean and variance moving average used in'
        ' batch normalization.')
    p.Define('causal_convolution', False,
             'Whether or not this is a causal convolution layer.')
    p.Define(
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    p.Define('bias', False, 'Whether or not to use bias.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GatedConvLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert p.activation == 'TANH' or p.activation == 'NONE', p.activation

    def CreateConvLayer(name):
      """Create a conv layer with the given name."""
      conv_p = ConvLayer.Params().Set(
          name=name,
          filter_shape=p.filter_shape,
          filter_stride=p.filter_stride,
          activation='NONE',
          batch_norm=p.batch_norm,
          weight_norm=p.weight_norm,
          bias=p.bias,
          bn_decay=p.bn_decay,
          causal_convolution=p.causal_convolution,
          params_init=p.params_init)
      self.CreateChild(name, conv_p)

    with tf.variable_scope(p.name):
      CreateConvLayer('activation')
      CreateConvLayer('gate')

  def FProp(self, theta, inputs, paddings):
    """Applies gated convolution to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor. It is expected to be of shape [batch, time,
          frequency, channel]. The time dimension corresponds to the height
          dimension as in images and the frequency dimension corresponds to the
          width dimension as in images.
      paddings: The paddings tensor. It is expected to be of shape [batch,
          time].
    Returns:
      outputs, out_paddings pair.
    """
    p = self.params
    activation, activation_padding = self.activation.FProp(
        theta.activation, inputs, paddings)
    gate = tf.sigmoid(self.gate.FProp(theta.gate, inputs, paddings)[0])
    if p.activation == 'TANH':
      activation = tf.tanh(activation)
    return activation * gate, activation_padding


class SpuProjectionLayer(base_layer.LayerBase):
  """Projection layer for spu infer, without bias, with tanh as activation."""

  @classmethod
  def Params(cls):
    """Params for SpuProjectionLayer."""
    p = super(SpuProjectionLayer, cls).Params()
    p.Define('input_dim', 0, 'Depth of the input.')
    p.Define('output_dim', 0, 'Depth of the output.')
    p.Define('input_min', -1.0, 'The min value of the input.')
    p.Define('input_max', 1.0, 'The max value of the input.')
    p.Define('input_quant_dtype', tf.quint16, 'The quantized dtype for input.')
    p.Define('weight_quant_dtype', tf.qint8, 'The quantized dtype for weight.')
    p.Define(
        'weight_norm', False,
        'If true, apply weight normalization to weights as proposed by'
        ' Salimans and Kingma, 2016: https://arxiv.org/abs/1602.07868')
    p.Define(
        'batch_norm', False,
        'Has to be False. Param introduced to be consistent with that'
        ' of ProjectionLayer.')
    p.Define(
        'has_bias', False,
        'Has to be False. Param introduced to be consistent with that'
        ' of ProjectionLayer.')
    p.Define(
        'activation', 'TANH',
        'Has to be TANH. Param introduced to be consistent with that'
        ' of ProjectionLayer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes the SpuProjectionLayer.

    Args:
      params: params used to construct SpuProjectionLayer.

    Raises:
      ValueError: p.name is not defined, or p.input_dim or p.output_dim is not
        positive, or p.weight_quant_dtype is not tf.qint8.
    """
    super(SpuProjectionLayer, self).__init__(params)
    p = self.params

    if not p.name:
      raise ValueError('name is not defined.')

    if p.input_dim <= 0 or p.output_dim <= 0:
      raise ValueError('input_dim %s and output_dim %s are not both positive.' %
                       (p.input_dim, p.output_dim))

    if p.weight_quant_dtype != tf.qint8:
      raise ValueError(
          'weight_quant_dtype %s is not tf.qint8.' % (p.weight_quant_dtype))

    w_shape = [p.input_dim, p.output_dim]
    w_pc = py_utils.WeightParams(
        shape=w_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)
      if p.weight_norm:
        g_pc = py_utils.WeightParams(
            shape=[p.output_dim],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('g', g_pc)
        w = (self.vars.g + 1.0) * tf.nn.l2_normalize(self.vars.w, [0])
      else:
        w = self.vars.w
      with tf.device(py_utils.SpuDevice()):
        self._quantized_w = py_utils.CreateQuantVariable(
            '%s_quantized_w' % p.name, w, p.weight_quant_dtype, w_shape)

  @property
  def spu_init_op(self):
    """Return the spu_init_op for projection layer."""
    return self._quantized_w.init

  def FProp(self, theta, inputs, paddings):
    """Apply projection to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers. Not used here.
      inputs: The inputs tensor, must be of p.input_quant_dtype and shape
        [batch_size, input_dim].
      paddings: The paddings tensor, of float type and shape [batch_size, 1].
    Returns:
      Output after applying projection and tanh.

    Raises:
      ValueError: 'inputs' dtype not match specified input_quant_dtype.
    """
    p = self.params

    if inputs.dtype != p.input_quant_dtype:
      raise ValueError('inputs dtype %s does not match input_quant_dype %s.' %
                       (inputs.dtype, p.input_quant_dtype))

    inputs_shape = tf.shape(inputs)
    ones = tf.ones(shape=inputs_shape, dtype=tf.float32)
    ones_q, _, _ = array_ops.quantize(ones, 0.0, 1.0, tf.quint16)
    with tf.name_scope(p.name):
      with tf.device(py_utils.SpuDevice()):
        # Convert 16-bit to 8-bit for 8-bit Matmul.
        if p.input_quant_dtype == tf.quint16:
          inputs = internal_math_ops.mul_n_clip(
              [inputs, ones_q], [p.input_min, 0.0], [p.input_max, 1.0],
              out_type=tf.quint8,
              output_min=p.input_min,
              output_max=p.input_max)
        activation_quant_dtype = (
            tf.qint32 if p.input_quant_dtype == tf.quint8 else tf.quint16)
        xw, _, _ = internal_math_ops.quantized_mat_mul_v2(
            a=inputs,
            b=self._quantized_w.var,
            min_a=p.input_min,
            max_a=p.input_max,
            weight_scale=self._quantized_w.scale,
            Toutput=activation_quant_dtype)
        out = internal_math_ops.quantized_tanh(xw, out_type=p.input_quant_dtype)
      # dequantize on cpu
      out = tf.dequantize(out, -1.0, 1.0)
      # zero out padded states.
      out *= (1.0 - paddings)
      return out


class NGramSampler(base_layer.LayerBase):
  """NGram sampler for performing latent sequence decomposition.

  Implements "Latent Sequence Decomposition" model as described in this paper:
  https://arxiv.org/pdf/1610.03035.pdf
  """

  @classmethod
  def Params(cls):
    p = super(NGramSampler, cls).Params()
    p.Define('max_ngram_len', 4, 'Max ngram length')
    p.Define('max_seq_len', None, 'Max sequence length')
    p.Define('vocab_size', None, 'The vocab size.')
    p.Define(
        'random_seed', None,
        'The random seed to use. Set it to non-zero value only for '
        'unit-test.')
    p.Define('target_sos_id', 1, 'Target start of sequence id.')
    p.Define('target_eos_id', 2, 'Target end of sequence id.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(NGramSampler, self).__init__(params)
    assert params.vocab_size
    assert params.max_seq_len
    self._global_states_inited = False

  def InitStates(self, all_ngrams, all_ngrams_length, all_ngrams_valid,
                 seq_lengths):
    """Initializes the global states of this sampler.

    This function should be called once and only once before the first call to
    SampleNgram is made.

    Args:
      all_ngrams: a tensor of shape [batch_size, max_seq_length, max_ngram_len].
          All valid ngrams across all possible positions.
      all_ngrams_length: must have the same shape as all_ngrams. Length of the
          corresponding ngram in 'all_ngrams'. Length of an ngram is measured by
          the number of underlying characters it represents. For example ngram
          'abc' has length 3.
      all_ngrams_valid: Must have the same shape as all_ngrams.
          "all_num_valid[i, j, k]" is 1 if the corresponding all_ngrams[i, j, k]
          is a valid ngram.
      seq_lengths: a vector of shape [batch_size]. Length of each of the
          sequences in number of characters.
    """
    p = self.params
    assert not self._global_states_inited
    self._global_states_inited = True
    self._all_ngrams = all_ngrams
    self._all_ngrams_length = all_ngrams_length
    self._all_ngrams_valid = all_ngrams_valid
    self._seq_lengths = seq_lengths
    self._seq_lengths = py_utils.with_dependencies([
        assert_shape_match(
            tf.shape(self._all_ngrams), tf.shape(self._all_ngrams_length)),
        assert_shape_match(
            tf.shape(self._all_ngrams), tf.shape(self._all_ngrams_valid)),
        py_utils.assert_equal(
            tf.shape(self._all_ngrams)[0],
            tf.shape(self._seq_lengths)[0]),
        py_utils.assert_equal(tf.shape(self._all_ngrams)[2], p.max_ngram_len)
    ], self._seq_lengths)

    self._cur_pos = inplace_ops.empty_like(self._seq_lengths, init=True)
    self._cur_ngrams = array_ops.deep_copy(self._all_ngrams[:, 0, :])
    self._cur_ngrams_length = array_ops.deep_copy(
        self._all_ngrams_length[:, 0, :])
    self._cur_ngrams_valid = array_ops.deep_copy(
        self._all_ngrams_valid[:, 0, :])

  def InitialNgram(self):
    p = self.params
    assert self._global_states_inited
    b_size = tf.shape(self._cur_pos)[0]
    sos = tf.fill([b_size], p.target_sos_id)
    weight = tf.ones([b_size], dtype=p.dtype)
    padding = tf.zeros([b_size], dtype=p.dtype)
    all_done = tf.constant(False, dtype=tf.bool)
    return (tf.stop_gradient(sos), tf.stop_gradient(weight),
            tf.stop_gradient(padding), tf.stop_gradient(all_done))

  def _GatherLogits(self, values, indices):
    """Gather logits for all candidate ngrams.

    Args:
      values: A matrix of floats of shape [batch_size, vocab_size].
      indices: A matrix of ints of shape [batch_size, #ngrams].

    Returns:
      A matrix of shape [batch_size, #ngrams].
    """
    values = py_utils.with_dependencies([
        py_utils.assert_equal(tf.shape(values)[0],
                              tf.shape(indices)[0]),
        py_utils.assert_equal(tf.rank(indices), 2)
    ], values)
    b_size = tf.shape(indices)[0]
    num_cols = tf.shape(indices)[1]
    indices = tf.expand_dims(indices, [-1])
    b_indices = tf.expand_dims(
        tf.tile(tf.reshape(tf.range(b_size), [-1, 1]), [1, num_cols]), [-1])
    return tf.gather_nd(values, tf.concat([b_indices, indices], 2))

  def SampleNgram(self, step_logits, annealing_temperature):
    """Does ngram sampling.

    Args:
      step_logits: a matrix of shape [batch_size, vocab_size].
      annealing_temperature: probability at which we simply sample ngram
          uniformly from the set of valid candidates.

    Returns:
      chosen_ngrams: a vector of shape [batch_size]. The set of chosen ngrams.
      chosen_weight: a vector of shape [batch_size].
      chosen_padding: a vector of shape [batch_size].
      all_done: True if we are done with all sequences.
    """
    p = self.params
    assert self._global_states_inited
    cur_ngram_logits = self._GatherLogits(step_logits, self._cur_ngrams)
    cur_ngram_logits = tf.cast(cur_ngram_logits, tf.float32)
    (chosen_ngrams, chosen_weight, chosen_padding, next_pos, next_ngrams,
     next_ngrams_length, next_ngrams_valid, all_done) = py_x_ops.sample_ngram(
         self._all_ngrams,
         self._all_ngrams_length,
         self._all_ngrams_valid,
         self._seq_lengths,
         tf.random_uniform([tf.shape(cur_ngram_logits)[0]], seed=p.random_seed),
         self._cur_ngrams,
         self._cur_ngrams_length,
         self._cur_ngrams_valid,
         cur_ngram_logits,
         self._cur_pos,
         annealing_temperature,
         max_ngram=p.max_ngram_len,
         eos_id=p.target_eos_id)
    self._cur_ngrams = next_ngrams
    self._cur_ngrams_length = next_ngrams_length
    self._cur_ngrams_valid = next_ngrams_valid
    self._cur_pos = next_pos
    if chosen_weight.dtype != p.dtype:
      chosen_weight = tf.cast(chosen_weight, p.dtype)
      chosen_padding = tf.cast(chosen_padding, p.dtype)
    return (tf.stop_gradient(chosen_ngrams), tf.stop_gradient(chosen_weight),
            tf.stop_gradient(chosen_padding), tf.stop_gradient(all_done))


class GumbelSoftmaxSampler(base_layer.LayerBase):
  """Gumbel Softmax sampler.

  Based on the softmax of the Gumbel sample distribution, as described in:

  https://arxiv.org/abs/1611.01144

  and

  https://arxiv.org/abs/1611.00712

  (also known as the Concrete distribution).
  """

  @classmethod
  def Params(cls):
    p = super(GumbelSoftmaxSampler, cls).Params()
    p.Define('input_dim', 0, 'Depth of the input.')
    p.Define(
        'temperature', None,
        'Temperature of the relaxation.  Values closer to 0 lead to more truly '
        'categorical-like samplers.  If temperature <= (input_dim - 1)^{-1}, '
        'then the underlying distribution PDF is log-convex w.r.t. its '
        'probabilities.')
    p.Define('seed', None, 'Sampling seed')
    p.Define(
        'one_hot_with_straight_through', False,
        'Use the argmax to get a single one-hot sample vector for FProp, but '
        'use a straight-through estimator on BProp.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GumbelSoftmaxSampler, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input_dim > 0
    assert p.temperature is not None, 'must provide a temperature > 0'
    p.temperature = tf.convert_to_tensor(p.temperature)

  def FProp(self, theta, inputs):
    """Get differentiable sample vectors from Gumbel-Softmax distributions.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The logits determining the log-probs of the underlying
        distribution batches.  Tensor of rank 2 or higher.  The last dimension
        corresponds to the number of classes.

    Returns:
      Tensor of the same type and shape as the input logits.  The right-most
      axis corresponds to the values of a single sample.  If temperature is
      very close to 0 or the property one_hot_with_straight_through is True,
      then each innermost column will be a one-hot vector.
    """
    p = self.params

    def GumbelNoise(logits):
      logits_shape = tf.shape(logits)
      uniform_samples = tf.random_uniform(
          shape=logits_shape,
          dtype=logits.dtype,
          seed=p.seed,
          name='uniform_samples')
      gumbel_noise = -tf.log(-tf.log(uniform_samples))
      return gumbel_noise

    with tf.name_scope(p.name):
      inputs = tf.convert_to_tensor(inputs, name='logits')
      inputs_static_shape = inputs.get_shape()
      assert inputs_static_shape.ndims is not None
      assert inputs_static_shape.ndims > 1
      assert inputs_static_shape[-1] == p.input_dim
      assert inputs.dtype == p.temperature.dtype

      @function.Defun()
      def GumbelSoftmaxGrad(logits, temperature, gumbel_noise, one_hots_dy):
        softmax_logits = (logits + gumbel_noise) / temperature
        softmax_val = tf.nn.softmax(softmax_logits)
        return tf.gradients(
            [softmax_val], [logits, temperature, gumbel_noise],
            grad_ys=[one_hots_dy])

      @function.Defun(grad_func=GumbelSoftmaxGrad)
      def GumbelSoftmaxSample(logits, temperature, gumbel_noise):
        if p.one_hot_with_straight_through:
          softmax_logits = logits + gumbel_noise
          gumbel_argmax = tf.argmax(softmax_logits,
                                    inputs_static_shape.ndims - 1)
          one_hot = tf.one_hot(
              gumbel_argmax, depth=p.input_dim, dtype=logits.dtype)
          return one_hot
        else:
          softmax_logits = (logits + gumbel_noise) / temperature
          return tf.nn.softmax(softmax_logits)

      centered_logits = tf.nn.log_softmax(inputs)
      gumbel_noise = GumbelNoise(centered_logits)
      gumbel_softmax_samples = GumbelSoftmaxSample(centered_logits,
                                                   p.temperature, gumbel_noise)
      gumbel_softmax_samples.set_shape(centered_logits.get_shape())

      return gumbel_softmax_samples


def Add(sums, deltas):
  """sums[i] += deltas[i] for all i."""
  return [tf.add_n([x, y]) for (x, y) in zip(sums, deltas)]


def _InplaceUpdate(t, acc, vals):
  """acc[i][t, :] = vals[i][t, :] for all i."""
  # t is an int64 vector.
  t = tf.cast(t, tf.int32)
  return [
      inplace_ops.alias_inplace_update(x, t, tf.expand_dims(y, 0))
      for (x, y) in zip(acc, vals)
  ]


def _EmptyLike(vals, init=True):
  """For each x in vals, allocates a same-sized tensor filled with zeros."""
  return [inplace_ops.empty_like(_, init=init) for _ in vals]


class LocalizedLabelSmoother(base_layer.LayerBase):
  """Smooths labels given as class ids.

  Implements the smoothing from https://arxiv.org/abs/1612.02695. Instead of
  1-hot class ids the model is trained to predict a distribution over classes
  that includes the correct class label and with a small probability the labels
  of tokens that appear nearby in time in the ground truth. This typically acts
  as a strong regularizer.

  """

  @classmethod
  def Params(cls):
    p = super(LocalizedLabelSmoother, cls).Params()
    p.Define('num_classes', 0, 'Number of classes')
    p.Define(
        'offsets', [], 'Offset (over time) for smoothing. At time T the '
        'smoothed target is class[T] + sum_i weights[i]*class[T+offset[i]]')
    p.Define('weights', [], 'Weight of the smoothing at corresponding offset')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LocalizedLabelSmoother, self).__init__(params)
    p = self.params
    assert p.num_classes > 0
    assert len(p.offsets) == len(p.weights)
    assert p.name

  def FProp(self, theta, target_paddings, target_labels, target_ids):
    """Convert class_ids to 1hot and smooth by neighborhood.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      target_paddings: float32 matrix [bs, seq_len]
      target_labels: int32 matrix [bs, seq_len]. This stores the target
        label output at each decoder step as generated by the speech input
        generator input_batch.tgt.labels
      target_ids: int32 matrix [bs, seq_len]. This stores the
        target_id that is fed to the decoder, as generated by the speech input
        generator input_batch.tgt.ids

    Returns:
      tensor [bs, seq_len, num_classes] denoting a smoothed
        distribution over num_classes
    """
    del target_ids  # Unused.
    p = self.params
    class_probabilities = tf.one_hot(
        target_labels, p.num_classes, dtype=FPropDtype(p))

    # Start list keeping the scaled class-probabilities at different offsets.
    output_distributions = [class_probabilities]
    seq_len = tf.shape(class_probabilities)[1]
    # If offsets < 0 we force a future output_act to be like a past token.
    # If offsets > 0 we force a past output_act to be like a future token.
    min_offset = np.min(p.offsets + [0])
    max_offset = np.max(p.offsets + [0])
    class_probabilities = tf.pad(class_probabilities,
                                 [[0, 0], [-min_offset, max_offset], [0, 0]])
    # Shift the weights to the left by one location - we don't make the
    # EOS more probable.
    class_weights = tf.pad(1.0 - target_paddings[:, 1:],
                           [[0, 0], [-min_offset, max_offset + 1]])
    class_weights = tf.expand_dims(class_weights, 2)

    for offset, weight in zip(p.offsets, p.weights):
      offset_in_padded = offset - min_offset
      output_distributions.append(
          class_probabilities[:, offset_in_padded:offset_in_padded + seq_len, :]
          * class_weights[:, offset_in_padded:offset_in_padded + seq_len, :] *
          weight)
    output_distributions = tf.add_n(output_distributions)
    output_distributions /= tf.reduce_sum(
        output_distributions, axis=-1, keepdims=True)
    return output_distributions


def GatedTanh(inputs, output_channels):
  """Returns tanh(x1) * sigmoid(x2).

  ... where x1 and x2 are two halves of the input split along the last dimension
  (channels), as described in the WaveNet paper.

  Args:
    inputs: the input tensor, of shape [..., output_channels * 2].
    output_channels: the number of channels in the output tensor.

  Returns:
    A tensor of shape [..., output_channels].
  """
  inputs = py_utils.with_dependencies([
      assert_shape_match([tf.shape(inputs)[-1]], [2 * output_channels]),
  ], inputs)
  x1, x2 = tf.split(inputs, 2, axis=-1)
  return tf.tanh(x1) * tf.sigmoid(x2)


class DilatedConvNet(base_layer.LayerBase):
  """Convolution layers with conditioning.

  This is designed to mimic the stack in
  google3/learning/deepmind/applications/tts/architectures/dila_tts2.py, which
  is also described in the WaveNet paper: https://arxiv.org/pdf/1609.03499.pdf.
  """

  @classmethod
  def Params(cls):
    p = super(DilatedConvNet, cls).Params()
    p.Define('num_layers', 10, 'Number of dilated convolution layers (N).')
    p.Define('filter_size', 3, 'Convolution filter size on the time dimension.')
    p.Define('causal_convolution', True, 'Use causal convolution.')
    p.Define('dilation_growth_rate', 2,
             'The growth factor of dilation_rate across layers.')
    p.Define('dilation_cycle_size', 10,
             'The number of dilations in each cycle.')
    p.Define('mix_output_channels', 64,
             'The number of output channels from the mixing unit.')
    p.Define('residual_channels', 384,
             'Number of channels in residual connections.')
    p.Define('input_channels', 1, 'Number of input channels.')
    p.Define('input_conv_filter_size', 4,
             'Convolution filter size on the time dimension.')
    p.Define('output_channels', 256, 'Number of output channels.')
    p.Define('num_output_layers', 2, 'Number of output 1x1 conv layers.')
    p.Define('use_skip_connections', True, 'Whether to use skip connections.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DilatedConvNet, self).__init__(params)
    p = self.params
    assert p.name
    assert p.num_layers > 0
    assert p.filter_size > 0
    assert p.input_channels > 0
    assert p.output_channels > 0
    assert p.mix_output_channels > 0
    assert p.residual_channels > 0

    with tf.variable_scope(p.name):
      self.CreateChild(
          'input_conv',
          self._BuildCausalConv1D(
              'input_conv',
              p.input_channels,
              p.residual_channels,
              filter_size=p.input_conv_filter_size))
      if p.use_skip_connections:
        self.CreateChild(
            'input_to_total',
            self._BuildCausalConv1D('input_to_total', p.residual_channels,
                                    p.output_channels))

      # Build the dilated conv layers.
      params_dilation_layers = []
      params_mix_to_residual_layers = []
      params_mix_to_total_layers = []
      for layer_index in range(p.num_layers):
        params_dilation_layers.append(
            self._BuildCausalConv1D(
                'dilation_%d' % layer_index,
                p.residual_channels,
                p.mix_output_channels * 2,
                filter_size=p.filter_size,
                dilation_rate=(p.dilation_growth_rate
                               **(layer_index % p.dilation_cycle_size))))
        if p.use_skip_connections:
          params_mix_to_total_layers.append(
              self._BuildCausalConv1D('mix_to_total_%d' % layer_index,
                                      p.mix_output_channels, p.output_channels))
        if layer_index != p.num_layers - 1 or not p.use_skip_connections:
          params_mix_to_residual_layers.append(
              self._BuildCausalConv1D('mix_to_residual_%d' % layer_index,
                                      p.mix_output_channels,
                                      p.residual_channels))
      self.CreateChildren('dilation', params_dilation_layers)
      if p.use_skip_connections:
        self.CreateChildren('mix_to_total', params_mix_to_total_layers)
      self.CreateChildren('mix_to_residual', params_mix_to_residual_layers)

      total_output_channels = p.output_channels
      if not p.use_skip_connections:
        total_output_channels = p.residual_channels
      params_out_layers = []
      for layer_index in range(p.num_output_layers):
        params_out_layers.append(
            self._BuildCausalConv1D('output_conv_%d' % layer_index,
                                    total_output_channels, p.output_channels))
        total_output_channels = p.output_channels
      self.CreateChildren('output_conv', params_out_layers)

  def _BuildCausalConv1D(self,
                         name,
                         input_channels,
                         output_channels,
                         filter_size=1,
                         dilation_rate=1):
    p = self.params
    conv_p = ConvLayer.Params()
    conv_p.name = name
    conv_p.params_init = p.params_init
    conv_p.causal_convolution = p.causal_convolution
    conv_p.filter_shape = (filter_size, 1, input_channels, output_channels)
    # Stride doesn't work with dilation, so we have to fix it at (1,1).
    conv_p.filter_stride = (1, 1)
    conv_p.dilation_rate = (dilation_rate, 1)
    conv_p.activation = 'NONE'
    # Disable batch_norm and use bias as WaveNet does.
    conv_p.batch_norm = False
    conv_p.bias = True
    return conv_p

  @property
  def conditioning_layers(self):
    return self.params.num_layers

  @property
  def conditioning_channels(self):
    return self.params.mix_output_channels * 2

  def FProp(self, theta, inputs, paddings, conditioning=None):
    """Feeds 'inputs' through the dilated convolution net.

    The computation goes as follows:
        [B, T, X, input_channels]
        |
        v
        input_conv (input_conv_filter_size x 1)
        |
        [B, T, X, residual_channels]
        |
        |---> input_to_total (1x1) ---> [B, T, X, output_channels]
        |
    for i in [0, num_layers) {
        |
    |---|
    |   v
    |   dilation (filter_size x 1 conv, with dilation)
    |   |
    |   [B, T, X, mix_output_channels * 2]
    |   |
    |   v
    |   conditioning (x += conditioning[i])
    |   |
    |   [B, T, X, mix_output_channels * 2]
    |   |
    |   v
    |   GatedTanh
    |   |
    |   [B, T, X, mix_output_channels]
    |   |
    |   |---> mix_to_total (1x1 conv) ---> [B, T, X, output_channels]
    |   |
    |   v
    |   mix_to_residual (1x1 conv)
    |   |
    |   [B, T, X, residual_channels]
    |   |
    |-->+
        |
        v
        (go back to dilation)
    }

        Sum({input,dilation}_to_total results) [B, T, X, output_channels]
        |
        v
        output_conv layers (1x1, with relu)
        |
        v
        [B, T, X, output_channels]

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a [Batch, Time, X, p.input_channels] tensor.
      paddings: a [Batch, Time] tensor.
      conditioning: a list of N [Batch, Time, X, conditioning_channels] tensors.

    Returns:
      A [Batch, Time, X, p.output_channels] tensor.
    """
    p = self.params
    with tf.name_scope(p.name):
      inputs = py_utils.HasShape(inputs, [-1, -1, -1, p.input_channels])
      res = self.input_conv.FProp(theta.input_conv, inputs, paddings)[0]
      if p.use_skip_connections:
        tot = self.input_to_total.FProp(theta.input_to_total, res, paddings)[0]
      for layer_index in range(p.num_layers):
        conv_out = self.dilation[layer_index].FProp(theta.dilation[layer_index],
                                                    res, paddings)[0]
        if conditioning:
          conv_out += conditioning[layer_index]
        mix_out = GatedTanh(conv_out, p.mix_output_channels)
        if p.use_skip_connections:
          tot += self.mix_to_total[layer_index].FProp(
              theta.mix_to_total[layer_index], mix_out, paddings)[0]
        if layer_index != p.num_layers - 1 or not p.use_skip_connections:
          res += self.mix_to_residual[layer_index].FProp(
              theta.mix_to_residual[layer_index], mix_out, paddings)[0]
        else:
          res = None
      if not p.use_skip_connections:
        tot = res
      for layer_index in range(p.num_output_layers):
        tot = tf.nn.relu(tot)
        tot = self.output_conv[layer_index].FProp(
            theta.output_conv[layer_index], tot, paddings)[0]
      return py_utils.with_dependencies([
          assert_shape_match(tf.shape(tot)[:-1],
                             tf.shape(inputs)[:-1]),
          assert_shape_match([tf.shape(tot)[-1]], [p.output_channels]),
      ], tot)


class UpsamplingConvNet(base_layer.LayerBase):
  """A convolution transpose net for upsampling along the time dimension."""

  @classmethod
  def Params(cls):
    p = super(UpsamplingConvNet, cls).Params()
    p.Define(
        'num_layers', 3, 'Number of convolution transpose layers. '
        'Every layer increases the time dim by a factor determined by the '
        'corresponding `layer_stride`.')
    p.Define('filter_size', 4,
             'Convolution filter shape along the time dimension.')
    p.Define('channels', 0, 'Number of input/output channels.')
    p.Define('layer_strides', [2] * p.num_layers,
             'Stride (upsampling rate) to use at each layer.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(UpsamplingConvNet, self).__init__(params)
    p = self.params
    assert p.name
    assert p.channels > 0
    assert len(p.layer_strides) == p.num_layers
    with tf.variable_scope(p.name):
      self.CreateChildren('conv', [
          ConvTransposeLayer.Params().Set(
              name='conv_%d' % layer_index,
              filter_shape=[p.filter_size, 1] + [p.channels] * 2,
              filter_stride=[p.layer_strides[layer_index], 1],
              params_init=p.params_init) for layer_index in range(p.num_layers)
      ])

  def FProp(self, theta, inputs, paddings):
    p = self.params
    with tf.name_scope(p.name):
      x = inputs
      for i, conv_layer in enumerate(self.conv):
        x, paddings = conv_layer.FProp(theta.conv[i], x, paddings)
        x = tf.nn.relu(x)
      return x, paddings


class FnLayer(base_layer.LayerBase):
  """A layer whose forward function is expressed as a stateful python callable.

  E.g.,
    ```
    # Suppose we want to use an external function Foo which contains
    # trainable variables.
    def Foo(x):
      return tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)

    # We want to use Foo in a babelfish model as a child layer.
    # In the parent layer's __init__, we can have:
    p = FnLayer.Params()
    p.name = 'foo'
    p.forward = Foo
    p.forward_arg_dtypes = [tf.float32]
    p.forward_arg_shapes = [(None, 100)]
    self.CreateChild('foo', p)
    ```
  """

  @classmethod
  def Params(cls):
    p = super(FnLayer, cls).Params()
    p.Define(
        'forward', None, 'A python callable, which takes '
        'a tuple of tf.Tensors and returns a tuple of tf.Tensors.')
    p.Define(
        'forward_arg_dtypes', [], 'A list of tf.DTypes indicating '
        'dtypes of the forward function args.')
    p.Define(
        'forward_arg_shapes', [], 'A list of tensor shapes indicating '
        'the expected shape of the forward function args.')
    return p

  @staticmethod
  def _Call(sig, *args):
    """Calls a tf function with the given signature."""
    print(type(sig))
    # pylint: disable=protected-access
    rets = function._call(sig, *args)[0]
    # pylint: enable=protected-access
    assert not isinstance(rets, tf.Operation)
    return rets

  @base_layer.initializer
  def __init__(self, params):
    super(FnLayer, self).__init__(params)

    p = self.params

    assert isinstance(p.forward_arg_dtypes, list), p.forward_arg_dtypes
    assert all(isinstance(_, tf.DType)
               for _ in p.forward_arg_dtypes), p.forward_arg_dtypes
    assert len(p.forward_arg_dtypes) == len(p.forward_arg_shapes)

    def CallForward(*args):
      """Calls the p.forward."""
      # Sanity checks of dtypes and shapes.
      assert len(args) == len(p.forward_arg_dtypes)

      # TODO(shlens, zhifengc): Consider reverting the tf.assert_equal hack
      # after we no longer need certain parts of the object detection code base.
      # pylint: disable=redefined-outer-name,g-import-not-at-top,reimported
      import tensorflow as tf
      # pylint: enable=redefined-outer-name,g-import-not-at-top,reimported
      saved_assert_equal = tf.assert_equal

      # pylint: disable=unused-argument,invalid-name
      def no_op(*args, **kwargs):
        return tf.no_op()

      # pylint: enable=unused-argument,invalid-name

      tf.assert_equal = no_op  # Make assert_equal a no op.

      for (x, dtype, shape) in zip(args, p.forward_arg_dtypes,
                                   p.forward_arg_shapes):
        assert x.dtype == dtype, ('%s vs. %s.', x, dtype)
        x.set_shape(shape)
      # Calls p.forward in a variable scope that allows reuse because
      # CallForward is called in both Fwd and Bak.
      with tf.variable_scope(p.name, reuse=tf.AUTO_REUSE):
        rets = p.forward(*args)

      tf.assert_equal = saved_assert_equal

      # Always return a list of tensors.
      if isinstance(rets, tuple):
        rets = list(rets)
      elif not isinstance(rets, list):
        rets = [rets]
      return rets

    def ConvertNoneGradientToZeros(xs, dxs):
      assert len(xs) == len(dxs)
      rets = []
      for (x, dx) in zip(xs, dxs):
        if dx is None:
          rets += [tf.zeros_like(x)]
        else:
          rets += [dx]
      return rets

    @function.Defun(*p.forward_arg_dtypes)
    def Fwd(*args):
      assert len(args) == len(p.forward_arg_dtypes)
      rets = CallForward(*args)
      self._private_vars.hidden = function.get_extra_vars()
      self._private_theta.hidden = function.get_extra_inputs()

      nvars, ntheta = len(self._private_vars.hidden), len(
          self._private_theta.hidden)
      # FProp assumes that vars.hidden[i] corresponds to theta.hidden[i].
      for i in range(min(nvars, ntheta)):
        v = self._private_vars.hidden[i]
        t = self._private_theta.hidden[i]
        # We assume that i-th theta is reading i-th variable.
        # TODO(zhifengc): We could have guaranteed that by passing a
        # custom variable getter in CallForward's variable scope,
        # which always adds an identity immediately after the variable
        # is created.
        assert t.name.endswith('/read:0'), (
            '%d %s vs. %s' % (i, v.name, t.name))
        assert v.name[:-2] == t.name[:-7], (
            '%d %s vs. %s' % (i, v.name, t.name))
      assert nvars == ntheta, ('%d vs. %d' % (nvars, ntheta))
      return rets

    # Analysis of the Fwd signature.  Fwd is a function of xs + theta
    # -> ys, where xs, theta and ys are lists of tensors.  theta are
    # implicitly captured during the construction of Fwd.
    xs_dtypes = p.forward_arg_dtypes
    theta_dtypes = [
        tf.DType(d.type)
        for d in Fwd.definition.signature.input_arg[len(xs_dtypes):]
    ]
    ys_dtypes = [tf.DType(d.type) for d in Fwd.definition.signature.output_arg]
    nxs, nys = len(xs_dtypes), len(ys_dtypes)

    def CheckDTypes(values, dtypes):
      assert len(values) == len(dtypes)
      for v, dtype in zip(values, dtypes):
        assert v is not None and isinstance(v, tf.Tensor), ('%s', v)
        assert v.dtype == dtype, ('%s is not %s', v, dtype)

    # Bak: xs + dys + theta -> dxs + dtheta
    @function.Defun(*(xs_dtypes + ys_dtypes))
    def Bak(*args):
      """Computes the gradients for Fwd."""
      xs, dys = list(args[:nxs]), list(args[nxs:])
      ys = CallForward(*xs)
      CheckDTypes(ys, ys_dtypes)
      # theta are placeholders corresponding to self._private_theta.hidden.
      theta = function.get_extra_args()
      # Fwd should have captured theta in the same order.
      CheckDTypes(theta, theta_dtypes)
      grads = tf.gradients(ys=ys, xs=xs + theta, grad_ys=dys)
      grads = ConvertNoneGradientToZeros(xs + theta, grads)
      CheckDTypes(grads, xs_dtypes + theta_dtypes)
      return grads

    # BakSwap computes the same as Bak with args rearanged:
    #   xs + theta + dys -> dxs + dtheta
    @function.Defun(*(xs_dtypes + theta_dtypes + ys_dtypes))
    def BakSwap(*args):
      xs, theta, dys = args[:nxs], args[nxs:-nys], args[-nys:]
      Bak.add_to_graph(tf.get_default_graph())
      return FnLayer._Call(Bak.definition.signature, *(xs + dys + theta))

    # Fwd's gradient function is BakSwap.
    Fwd.set_grad_func(BakSwap)

    # Because function._call does not add the function definitions to
    # graph def, we need to explicitly do that there.
    Fwd.add_to_graph(tf.get_default_graph())
    self._func_sig = Fwd.definition.signature

  def FProp(self, theta, *args):
    p = self.params
    for (x, dtype, shape) in zip(args, p.forward_arg_dtypes,
                                 p.forward_arg_shapes):
      assert x.dtype == dtype, ('%s vs. %s.', x, dtype)
      x.shape.assert_is_compatible_with(shape)
    return FnLayer._Call(self._func_sig, *(list(args) + theta.hidden))


class DeterministicFeedForwardNet(FeedForwardNet):
  """Deterministic FeedForwardNet for correct rematerialization."""

  def FProp(self, theta, inputs, paddings=None):
    p = self.params
    num_layers = len(self.fc)
    activation = p.activation
    if isinstance(activation, six.string_types):
      activation = [activation] * num_layers
    else:
      assert len(activation) == num_layers

    dropout_prob = p.dropout_prob
    if isinstance(dropout_prob, (list, tuple)):
      assert len(dropout_prob) == num_layers
    else:
      dropout_prob = [dropout_prob] * num_layers

    in_dim, layer_in = p.input_dim, inputs
    prev_proj_out = None
    for i in range(num_layers):
      layer_in = py_utils.with_dependencies(
          [py_utils.assert_shape_match([tf.shape(layer_in)[-1]], [in_dim])],
          layer_in)
      out_dim = p.hidden_layer_dims[i]
      layer_out = self.fc[i].FProp(theta.fc[i], layer_in, paddings)
      skip_connection = self._skip_connections[i]
      if skip_connection == 'ResNet' and prev_proj_out is not None:
        layer_out = tf.add(prev_proj_out, layer_out)
      prev_proj_out = layer_out
      layer_out = self.bn[i].FProp(theta.bn[i], layer_out, paddings)
      if activation[i] != 'NONE':
        activations_functions = {
            'RELU': tf.nn.relu,
            'RELU6': tf.nn.relu6,
            'SIGMOID': tf.sigmoid,
            'TANH': tf.tanh,
        }
        layer_out = activations_functions[activation[i]](layer_out)
      if dropout_prob[i] > 0.0 and (not p.is_eval or p.dropout_at_eval):
        seeds = py_utils.GetOpSeedPair()
        layer_out = py_utils.DeterministicDropout(layer_out,
                                                  1.0 - dropout_prob[i], seeds)
      if skip_connection == 'DenseNet':
        layer_in = tf.concat([layer_in, layer_out], axis=-1)
      else:
        layer_in = layer_out
      in_dim = out_dim
    return layer_in


def TfEditDistance(ref_tokens, ref_paddings, hyp_tokens, hyp_paddings, seq_len):
  """Compute the edit distance between 'ref' and 'hyp'.

  Args:
    ref_tokens: tf.int32 tensor of shape [batch_size, seq_len], the ground-truth
        tokens.
    ref_paddings: paddings tensor, of shape [batch_size, seq_len].
    hyp_tokens: tf.int32 tensor of shape [batch_size, seq_len], the
        hyp tokens.
    hyp_paddings: paddings tensor for hyp.
    seq_len: int, the max sequence lenrefh.
  Returns:
    A pair (distance, table), where 'distance' is a tensor of shape [batch_size]
        containing the edit distances between 'ref_tokens' and
        'hyp_tokens', and 'table' is a tensor of shape [batch_size,
        seq_len, seq_len] which is the table filled during dynamic programming.
        'table[i, x, y]' represents the edit distance between the first x
        ref_tokens and the first y hyp_tokens for the i-th pair.
  """

  # At a high level, we try to fill a 2d table with a double for loop, top down
  # and left to right. Rows in the table correspond to the ref tokens
  # and columns correspond to the hyp tokens.

  ref_tokens = tf.transpose(ref_tokens)
  hyp_tokens = tf.transpose(hyp_tokens)
  assert ref_tokens.dtype == tf.int32
  assert hyp_tokens.dtype == tf.int32

  def FillOneRow(hyp, prev_row, ref_i, i):
    """Fill one row in the table.

    Args:
      hyp: The tensor of hyp tokens, of shape [seq_len, batch_size].
      prev_row: The previous row, of shape [seq_len, batch_size]
      ref_i: ref tokens corresponding to the current row, of shape
          [batch_size].
      i: a tf.int32 scalar, the current row index.
    Returns:
      A newly filled row of shape [seq_len, batch_size]
    """

    def FillCell(theta, state0, inputs):
      """Fill one table cell.

      Args:
        theta: A NestedMap of params used across rnn steps. Here it contains
            ref_i tensor.
        state0: The start state for this rnn step.
        inputs: Other input tensors. Here it contains the cell values to the
            left, above and left-above of the current cell to be filled in.
      Returns:
        A pair of NestedMaps. The first map contains the new cell value, and the
        second NestedMap is empty.
      """
      above = tf.cast(inputs.above, tf.int32)
      diag = tf.cast(inputs.diag, tf.int32)
      hyp_j = tf.cast(inputs.hyp, tf.int32)
      ref_i = tf.cast(theta.ref_i, tf.int32)
      left = tf.cast(state0.cell, tf.int32)
      assert above.dtype == tf.int32, above.dtype
      assert diag.dtype == tf.int32, diag.dtype
      assert left.dtype == tf.int32, left.dtype
      assert ref_i.dtype == tf.int32, ref_i.dtype
      assert hyp_j.dtype == tf.int32, hyp_j.dtype

      # "above + 1" is the edit distance if to DEL ref_tokens[i].
      # "left + 1" is the edit distance if to INSERT hyp_tokens[j]
      # "diag + 1" is the edit distance if to SUB ref_tokens[i] ->
      #   hyp_tokens[j].
      cell = tf.minimum(
          tf.minimum(diag + tf.cast(tf.not_equal(ref_i, hyp_j), tf.int32),
                     above + 1), left + 1)
      return py_utils.NestedMap(cell=cell), py_utils.NestedMap()

    theta = py_utils.NestedMap(ref_i=ref_i)
    # This is the edit distance between "" and hyp[0:i+1]
    state0 = py_utils.NestedMap(cell=tf.zeros_like(ref_i) + i + 1)
    # diag[j] represents the edit distance between hyp_tokens[0:j]
    # and ref_tokens[0:i]
    inputs = py_utils.NestedMap(
        above=prev_row,
        diag=tf.concat(
            [tf.expand_dims(tf.zeros_like(ref_i), 0) + i, prev_row[:-1]], 0),
        hyp=hyp)
    cumulated, _ = recurrent.Recurrent(theta, state0, inputs, FillCell)
    return cumulated.cell

  def FillRowHelper(theta, state0, inputs):
    """Fill a row in the dynamic programming table.

    This is mostly a wrapper around FillOneRow above.

    Args:
      theta: A NestedMap of params used across rnn steps. Here it contains
          the hyp tensor.
      state0: The start state for this rnn step. Here it contains the previous
          row in the table that has already been filled.
      inputs: Other input tensors. Here it contains the ref token at the current
          position.
    Returns:
      A pair of NestedMaps. The first map contains a newly filled row, and the
      second NestedMap is empty.
    """
    i = state0.i
    prev_row = state0.row
    ref_i = tf.cast(inputs.ref, tf.int32)
    hyp = theta.hyp

    assert hyp.dtype == tf.int32, hyp.dtype
    assert i.dtype == tf.int32, i.dtype
    assert prev_row.dtype == tf.int32, prev_row.dtype
    assert ref_i.dtype == tf.int32, ref_i.dtype

    new_row = FillOneRow(hyp, prev_row, ref_i, i)
    return py_utils.NestedMap(i=i + 1, row=new_row), py_utils.NestedMap()

  theta = py_utils.NestedMap(hyp=hyp_tokens)
  inputs = py_utils.NestedMap(ref=ref_tokens)
  batch_size = tf.shape(ref_tokens)[1]
  # Initialize row with [1, 2, ..., seq_len], such that row[j] represents
  # the edit distance between the first j+1 hyp_tokens and "".
  state0 = py_utils.NestedMap(
      i=tf.constant(0, tf.int32),
      row=tf.tile(tf.expand_dims(tf.range(seq_len) + 1, 1), [1, batch_size]))

  cumulated, _ = recurrent.Recurrent(theta, state0, inputs, FillRowHelper)
  # [batch, ref, hyp].
  table = tf.transpose(cumulated.row, [2, 0, 1])
  # first_row corresponds to the edit distance between '' and
  # 'hyp_tokens'
  first_row = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(seq_len) + 1, 0), 1),
      [batch_size, 1, 1])
  # first_column corresponds to the edit distance between '' and
  # 'ref_tokens'
  first_column = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(seq_len + 1), 0), 2),
      [batch_size, 1, 1])
  table = tf.concat([first_column, tf.concat([first_row, table], 1)], 2)

  # NOTE(yonghui): here we assume paddings can only appear at the end of a
  # sequence.
  ref_lens = tf.cast(tf.reduce_sum(1.0 - ref_paddings, 1), tf.int32)
  hyp_lens = tf.cast(tf.reduce_sum(1.0 - hyp_paddings, 1), tf.int32)
  gather_indices = tf.stack([tf.range(batch_size), ref_lens, hyp_lens], 1)
  dist = tf.gather_nd(table, gather_indices)

  return dist, table


def _ToTuple(x):
  return x if isinstance(x, tuple) else (x,)


class SequentialLayer(base_layer.LayerBase):
  """A layer which connects a few layers in a sequence."""

  @classmethod
  def Params(cls):
    p = super(SequentialLayer, cls).Params()
    p.Define('sub', [], 'A list of layers\' params.')
    p.Define('repeat', 1, 'Repeat layers specified in \'sub\' '
             'this many times.')
    p.Define(
        'nary', None, 'If not None, ensures the output has these many '
        'tensors. None(s) are returned if there aren\'t enough return '
        'values to return. Otherwise, some return values are dropped.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SequentialLayer, self).__init__(params)
    p = self.params
    assert p.name
    with tf.variable_scope(p.name):
      if p.repeat <= 1:
        self._seq = []
        for sub in p.sub:
          assert sub.name
          name = sub.name
          child = sub.cls(self.CopyBaseParams(p, sub.Copy()))
          self.AddChild(name, child)
          self._seq.append((name, child))
      else:
        # We create 'repeat' number of sub layers. Each sub layer is a
        # sequential layer specified by 'sub'.  This allows us to name each
        # repetition with a unique name.
        children = []
        for i in range(p.repeat):
          children.append(p.Copy().Set(name='%03d' % i, repeat=1))
        self.CreateChildren('rep', children)

  def FProp(self, theta, *args):
    p = self.params
    tf.logging.vlog(1, 'layer %s', self.params.name)
    if p.repeat <= 1:
      for (name, ch) in self._seq:
        th = theta[name]
        args = _ToTuple(args)
        tf.logging.vlog(1, '  call %s %s %d %s', ch.params.name, ch, len(args),
                        str(args))
        args = ch.FProp(th, *args)
    else:
      for (ch, th) in zip(self.rep, theta.rep):
        args = _ToTuple(args)
        tf.logging.vlog(1, '  call %s %s %d %s', ch.params.name, ch, len(args),
                        str(args))
        args = ch.FProp(th, *args)

    # Ensures the return value has length of p.nary.
    p = self.params
    if p.nary is not None:
      if isinstance(args, tuple):
        if len(args) < p.nary:
          args += (None,) * (p.nary - len(args))
        else:
          args = args[:p.nary]
      elif isinstance(args, list):
        if len(args) < p.nary:
          args += [None] * (p.nary - len(args))
        else:
          args = args[:p.nary]
      else:
        assert 1 <= p.nary
        args = (args,) + (None,) * (p.nary - 1)

    return args

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0
    for _ in range(p.repeat):
      for sub in p.sub:
        tf.logging.vlog(1, '  seq abs fprop %s %s %d %s', sub.name, sub.cls,
                        len(args), str(args))
        meta = sub.cls.FPropMeta(sub, *args)
        py_utils.CheckShapes(meta.out_shapes)
        total += meta.flops
        args = meta.out_shapes
    return py_utils.NestedMap(flops=total, out_shapes=args)


class ParallelLayer(base_layer.LayerBase):
  """A layer which connects a few layers in a parallel."""

  @classmethod
  def Params(cls):
    p = super(ParallelLayer, cls).Params()
    p.Define(
        'sub', [], 'A list of layers\' params. Each layer\'s '
        'FProp must return one Tensor or a tuple of Tensors. '
        'Their return values then can be merged according to the '
        'merge method. ')
    p.Define(
        'merge', None, 'Method to combine sub-layers\' outputs.'
        'It must be a callable list(tuple(tf.Tensor)) -> tuple(tf.Tensor).')
    p.Define(
        'merge_meta', None, 'Callable to compute the meta of merge(). It '
        'takes a list of tuples of TensorShape, and returns a NestedMap with '
        'flops and out_shapes, etc.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ParallelLayer, self).__init__(params)
    p = self.params
    assert p.name
    self._seq = []
    with tf.variable_scope(p.name):
      for sub in p.sub:
        assert sub.name
        name = sub.name
        child = sub.cls(self.CopyBaseParams(p, sub.Copy()))
        self.AddChild(name, child)
        self._seq.append((name, child))

  def FProp(self, theta, *args):
    p = self.params

    # Computes sub layers in parallel.
    outputs = []
    for (name, ch) in self._seq:
      th = theta[name]
      out = ch.FProp(th, *args)
      if isinstance(out, (list, tuple)):
        outputs.append(tuple(out))
      else:
        outputs.append((out,))
    rets = p.merge(outputs)
    return rets if len(rets) > 1 else rets[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    py_utils.CheckShapes(args)
    total = 0
    outputs = []
    for sub in p.sub:
      tf.logging.vlog(1, '  par abs fprop %s %s %d %s', sub.name, sub.cls,
                      len(args), str(args))
      meta = sub.cls.FPropMeta(sub, *args)
      py_utils.CheckShapes(meta.out_shapes)
      tf.logging.vlog(1, '  par abs fprop %s %s %d %s %s', sub.name, sub.cls,
                      len(args), str(args), meta.DebugString())
      total += meta.flops
      outputs.append(meta.out_shapes)

    meta = p.merge_meta(outputs)
    py_utils.CheckShapes(meta.out_shapes)
    meta.flops += total
    return meta


class MapLayer(base_layer.LayerBase):
  """A layer applies a lambda on every argument."""

  @classmethod
  def Params(cls):
    p = super(MapLayer, cls).Params()
    p.Define('fn', None, 'A callable tensor->tensor.')
    p.Define('fn_meta', None, 'A callable shape->(flops, shape).')
    p.Define('kwargs', {}, 'Keyword arguments to fn.')
    return p

  def FProp(self, theta, *args):
    """Applies lambda(x, *kwargs) for every non-None arg."""
    del theta
    p = self.params
    ret = [None if x is None else p.fn(x, **p.kwargs) for x in args]
    return tuple(ret) if len(ret) > 1 else ret[0]

  @classmethod
  def FPropMeta(cls, p, *args):
    flops, rets = 0, []
    for x in args:
      if x is None:
        rets.append(None)
      else:
        cost, shape = p.fn_meta(x)
        py_utils.CheckShapes((shape,))
        flops += cost
        rets.append(shape)
    return py_utils.NestedMap(flops=flops, out_shapes=tuple(rets))


class LinearLayer(base_layer.LayerBase):
  """Linear layer."""

  @classmethod
  def Params(cls):
    p = super(LinearLayer, cls).Params()
    p.Define('input_dims', 0, 'Depth of the input.')
    p.Define('output_dims', 0, 'Depth of the output.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LinearLayer, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      self.CreateVariable(
          'w',
          py_utils.WeightParams(
              shape=[p.input_dims, p.output_dims],
              init=p.params_init,
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  def FProp(self, theta, inputs):
    """Apply projection to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    p = self.params
    act = tf.matmul(tf.reshape(inputs, [-1, p.input_dims]), theta.w)
    act = tf.reshape(
        act, tf.concat([tf.shape(inputs)[:-1], [p.output_dims]], axis=0))
    return act

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    dims = inputs.as_list()
    assert p.input_dims == dims[-1]
    # c_{ij} += x_{ik} * y_{kj} are considered 2 flops.
    return py_utils.NestedMap(
        flops=np.prod(dims) * p.output_dims * 2,
        out_shapes=(tf.TensorShape(dims[:-1] + [p.output_dims]),))


class BiasLayer(base_layer.LayerBase):
  """Bias layer."""

  @classmethod
  def Params(cls):
    p = super(BiasLayer, cls).Params()
    p.Define('dims', 0, 'Depth of the input.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BiasLayer, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      self.CreateVariable(
          'b',
          py_utils.WeightParams(
              shape=[p.dims],
              init=p.params_init,
              dtype=p.dtype,
              collections=[self.__class__.__name__ + '_vars']))

  def FProp(self, theta, inputs):
    """Adds bias to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: The inputs tensor.  Shaped [..., dims].

    Returns:
      Inputs plus bias.
    """
    return inputs + theta.b

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    dims = inputs.as_list()
    assert dims[-1] == p.dims
    return py_utils.NestedMap(flops=np.prod(dims), out_shapes=(inputs,))


class Conv2DLayerNoPadding(base_layer.LayerBase):
  """2-D Convolution layer w/o padding."""

  @classmethod
  def Params(cls):
    p = super(Conv2DLayerNoPadding, cls).Params()
    p.Define(
        'filter_shape', (0, 0, 0, 0),
        'Filter shape. Must be a sequence of length 4. Elements are in'
        ' the order of height (time), width (frequency), in_channel,'
        ' out_channel. ')
    p.Define(
        'filter_stride', (0, 0),
        'Filter stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the height dimension. The second int'
        ' specifies the stride on the width dimension.')
    p.Define(
        'dilations', (1, 1), ' An optional list of ints. Defaults to [1, 1]. '
        '1-D tensor of length 2. The dilation factor for each dimension '
        'of input. If set to k > 1, there will be k-1 skipped cells '
        'between each filter element on that dimension.')
    p.Define('padding', 'SAME', 'SAME|VALID')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(Conv2DLayerNoPadding, self).__init__(params)
    p = self.params
    assert p.name
    assert p.padding in ['SAME', 'VALID']
    assert len(p.filter_shape) == 4
    assert len(p.filter_stride) == 2
    assert len(p.dilations) == 2
    assert all(x > 0 for x in p.filter_shape)
    assert all(x > 0 for x in p.filter_stride)
    w_pc = py_utils.WeightParams(
        shape=p.filter_shape,
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)

  def FProp(self, theta, x):
    """Apply convolution to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      x: The inputs tensor. It is expected to be of shape [batch,
        height, width, channel].

    Returns:
      Convolution output.
    """
    p = self.params
    with tf.name_scope(p.name):
      return tf.nn.conv2d(
          input=x,
          filter=theta.w,
          strides=[1, p.filter_stride[0], p.filter_stride[1], 1],
          padding=p.padding,
          dilations=[1, p.dilations[0], p.dilations[1], 1],
          data_format='NHWC')

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    b, h, w, c = inputs.as_list()
    fh, fw, ic, oc = p.filter_shape
    assert ic == c
    sh, sw = p.filter_stride
    flops = b * h * w * fh * fw * ic * oc * 2  # mul/add counts as 2 flop.
    if p.padding == 'SAME':
      outputs = tf.TensorShape([b, (h + sh - 1) // sh, (w + sw - 1) // sw, oc])
    else:
      out_height = (h - fh) // sh + 1
      out_width = (w - fw) // sw + 1
      outputs = tf.TensorShape([b, out_height, out_width, oc])
    return py_utils.NestedMap(flops=flops, out_shapes=(outputs,))


class Pool2DLayer(base_layer.LayerBase):
  """Pooling layer, by default performs max-pooling."""

  @classmethod
  def Params(cls):
    p = super(Pool2DLayer, cls).Params()
    p.Define(
        'window', (0, 0),
        'Window shape. Must be a pair of ints. Elements are in'
        ' the order of height (time), width (frequency).')
    p.Define(
        'stride', (0, 0),
        'Window stride to use. Must be a pair of ints. The first int'
        ' specifies the stride on the time dimension. The second int'
        ' specifies the stride on the frequency dimension.')
    p.Define('type', 'MAX', 'Pooling type: MAX|AVG')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(Pool2DLayer, self).__init__(params)
    p = self.params
    assert p.name
    assert len(p.window) == 2
    assert len(p.stride) == 2
    assert all([x > 0 for x in p.window])
    assert all([x > 0 for x in p.stride])
    assert p.type in ['MAX', 'AVG']

  def FProp(self, theta, x):
    """Apply pooling to inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      x: The inputs tensor. It is expected to be of shape [batch, height,
        width, channel].

    Returns:
      Pooling results.
    """
    p = self.params
    with tf.name_scope(p.name):
      return tf.nn.pool(
          input=x,
          window_shape=p.window,
          pooling_type=p.type,
          strides=p.stride,
          padding='SAME',
          data_format='NHWC')

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    b, h, w, c = inputs.as_list()
    wh, ww = p.window
    sh, sw = p.stride
    outputs = [b, (h + sh - 1) // sh, (w + sw - 1) // sw, c]
    # Assume a comparison (pool) is 1 flop.
    flops = np.prod(outputs) * (wh * ww)
    return py_utils.NestedMap(
        flops=flops, out_shapes=(tf.TensorShape(outputs),))


class FetchLayer(base_layer.LayerBase):
  """A layer facilitating fetching activations."""

  @base_layer.initializer
  def __init__(self, params):
    super(FetchLayer, self).__init__(params)
    assert self.params.name
    self._activations = None

  @classmethod
  def FPropMeta(cls, params, *args):
    return py_utils.NestedMap(flops=0, out_shapes=args)

  @property
  def activation(self):
    rets = self._activations
    assert rets is not None
    assert isinstance(rets, tuple)
    return rets if len(rets) > 1 else rets[0]

  def FProp(self, theta, *args):
    del theta
    # FProp was called twice when FetchLayer is included in StackedRecurrent
    if self._activations is None:
      self._activations = args
    return args
