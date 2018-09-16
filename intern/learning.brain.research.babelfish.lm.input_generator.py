"""Language model data module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google3.learning.brain.research.babelfish import base_input_generator
from google3.learning.brain.research.babelfish import py_utils
from google3.learning.brain.research.babelfish import tokenizers
from google3.learning.brain.research.babelfish.ops import py_x_ops


class LmInput(base_input_generator.BaseSequenceInputGenerator):
  """Generator for NMT example sstable."""

  @classmethod
  def Params(cls):
    """Defaults params for NmtInput."""
    p = super(LmInput, cls).Params()
    p.Define(
        'proto', 'string', 'Input sstable values are of this proto type.'
        'Current supported: string, '
        'speech.DataUtterance:trans.transcript_truth, '
        'speech.DataUtterance:trans.transcript')
    p.Define('domain_list', [],
             'List of strings corresponding to valid domain values.')
    p.Define(
        'domain_weights', [],
        'List of floats corresponding to domain weights.  domain_weights[i] is '
        'the weight that should be assigned to each token of each example from '
        'domain_list[i].')
    p.Define('gold_chunks', False, 'Flag to include chunking tags in input')
    p.tokenizer = tokenizers.WordPieceModel.Params()
    return p

  @staticmethod
  def GetChunks(ids, labels, paddings):
    # TODO(jmluo) this happens before trimming and transposing
    py_utils.HasRank(ids, 2)
    # separate BIO tags from true ids
    tags = ids[:, 0::2] # Note that this also includes <S>
    ids = tf.concat([ids[:, 0:1], ids[:, 1:-1:2]], axis=1)
    # adjust labels accordingly
    labels = labels[:, 0::2]
    paddings = paddings[:, 0::2]

    # compute chunk ids
    is_B = tf.equal(tags, 4)
    is_I = tf.equal(tags, 5)
    is_O = tf.equal(tags, 6)
    is_BI = tf.logical_or(is_B, is_I)
    chunk_ids = tf.cumsum(tf.to_int32(is_B), axis=1) * tf.to_int32(is_BI) # shouldn't have overflow issues here
    # is_BO = tf.logical_or(is_B, is_O)
    # last_word_marks = tf.logical_and(is_BI, tf.logical_not(tf.concat([is_I[:, 1:], tf.zeros([tf.shape(ids)[0], 1], dtype=tf.bool)], axis=1)))
    # # last_word_marks => chunk_ids
    # # tf.assert_equal(tf.logical_or(tf.logical_not(last_word_marks), tf.greater(chunk_ids, 0)), tf.ones_like(chunk_ids, dtype=tf.bool))
    # # have the same number of chunks
    # last_word_marks = tf.to_int32(last_word_marks)
    # tf.assert_equal(tf.reduce_max(chunk_ids, axis=1), tf.reduce_sum(last_word_marks, axis=1))
    return ids, labels, paddings, chunk_ids #(chunk_ids, last_word_marks)

  def __init__(self, params):
    params.pad_to_max_seq_length = True
    super(LmInput, self).__init__(params)
    p = self.params

    assert len(p.domain_list) == len(p.domain_weights)

    self._text, self._word_count, self._domain = self._BuildDataSource()

    self._ids, self._labels, self._paddings = self.StringsToIds(self._text)

    # deal with extra chunking tags in input
    if p.gold_chunks:
      self._ids, self._labels, self._paddings, self._chunk_ids = LmInput.GetChunks(self._ids, self._labels, self._paddings)

    self._input_batch_size = tf.shape(self._ids)[0]
    tf.summary.histogram('examples/sequence_length',
                         tf.reduce_sum(1.0 - self._paddings, axis=1))
    # Make a 1-hot vector of size [batch, len(p.domain_list)].
    if p.domain_list:
      self._domain_embedding = tf.one_hot(
          self._domain, depth=len(p.domain_list))
      # Domain embedding is 1-hot so, for each batch element, this product is
      # nonzero only in the index corresponding to the domain.
      # [batch, domains] x [domains] --> [batch, domains] --> [batch]
      token_weight = tf.reduce_sum(self._domain_embedding * p.domain_weights, 1)
      # [batch] --> [batch, 1] --> [batch, len]
      max_seq_len = tf.shape(self._ids)[1]
      token_weight = tf.tile(tf.expand_dims(token_weight, 1), [1, max_seq_len])
    else:
      self._domain_embedding = tf.fill(
          [self._input_batch_size, len(p.domain_list)], -1)
      token_weight = tf.constant(1.0)

    self._weights = (1.0 - self._paddings) * token_weight

    if py_utils.use_tpu():
      # When flush_every_n is on, at end of each epoch, our input
      # generator can generate a batch smaller than
      # bucket_batch_limit.
      assert not p.flush_every_n, 'flush_every_n is not allowed on TPU.'
      assert min(self.scaled_bucket_batch_limit) == max(
          self.scaled_bucket_batch_limit)
      bs = min(self.scaled_bucket_batch_limit)

      def SetShape(x):
        x.set_shape([bs, p.target_max_length])

      SetShape(self._ids)
      SetShape(self._labels)
      SetShape(self._paddings)
      SetShape(self._weights)
      self._word_count.set_shape([bs])
      self._domain_embedding.set_shape([bs, len(p.domain_list)])

  def _DataSourceFromFilePattern(self, file_pattern):
    p = self.params
    normalization = 'lowercase'
    if hasattr(p.tokenizer, 'normalization'):
      normalization = p.tokenizer.normalization
    return py_x_ops.lm_text_input(
        file_pattern=file_pattern,
        normalization=normalization,
        proto=p.proto,
        wpm=(p.tokenizer.wpm_model
             if p.tokenizer.cls is tokenizers.WordPieceModel else ''),
        domain_list=p.domain_list,
        **self.CommonInputOpArgs())

  def InputBatch(self):
    ret = py_utils.NestedMap()
    ret.ids = self._ids
    ret.labels = self._labels
    ret.paddings = self._paddings
    ret.weights = self._weights
    if not py_utils.use_tpu():
      ret.text = self._text
    ret.word_count = self._word_count
    ret.domain_embedding = self._domain_embedding

    if self.params.gold_chunks:
      ret.chunk_ids = self._chunk_ids

    return ret


class FixedSizeRandomLmInput(base_input_generator.BaseSequenceInputGenerator):
  """A test LM input generator which generates fix-sized tensors."""

  def __init__(self, params):
    super(FixedSizeRandomLmInput, self).__init__(params)

    p = self.params

    bs = max(self.scaled_bucket_batch_limit)
    slen = max(self.scaled_bucket_batch_limit)

    batch = py_utils.NestedMap()
    batch.ids = tf.to_int32(
        tf.random_uniform(shape=[bs, slen], maxval=p.tokenizer.vocab_size))
    batch.labels = tf.concat(
        [
            batch.ids[:, 1:],
            tf.fill([bs, 1], tf.constant(self.tokenizer.eos_id, dtype=tf.int32))
        ],
        axis=1)
    batch.paddings = tf.zeros(tf.shape(batch.ids))
    batch.weights = 1 - batch.paddings
    batch.text = tf.fill([bs], 'random text')
    batch.word_count = tf.fill([bs], slen)

    self._input_batch_size = tf.constant(bs, dtype=tf.int32)
    self._batch = batch

  def InputBatch(self):
    return self._batch


def SpeechInputAdapter(input_batch):
  """Convert input_batch to a format compatible with what's from ...

  lm.input_generator.LmInput. It is expected that 'input_batch' is from one of
  the speech InputGenerators, e.g. Greco3Input or KaldiInput

  Args:
    input_batch: a NestedMap. The original input_batch from speech
        InputGenerators.
  Returns:
    A NestedMap with the same structure as that from LmInput.InputBatch().
  """
  ret = py_utils.NestedMap
  ret.ids = input_batch.tgt.ids
  ret.labels = input_batch.tgt.labels
  ret.paddings = input_batch.tgt.paddings
  ret.weights = input_batch.tgt.weights
  ret.text = input_batch.tgt.transcripts
  # Just some notion of the length of the sequence.
  ret.word_count = tf.cast(
      tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1), tf.int32)
  # Some notion of the domain of size [batch].
  ret.domain_embedding = tf.fill(tf.shape(ret.ids)[:1], -1)
  return ret
