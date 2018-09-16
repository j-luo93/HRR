"""Train word-level LMs on PTB data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google3.learning.brain.research.babelfish import base_model_params
from google3.learning.brain.research.babelfish import layers
from google3.third_party.tensorflow_lingvo.core import lr_schedule
from google3.learning.brain.research.babelfish import model_registry
from google3.learning.brain.research.babelfish import optimizer
from google3.learning.brain.research.babelfish import py_utils
from google3.third_party.tensorflow_lingvo.core import tokenizers
from google3.learning.brain.research.babelfish.lm import input_generator as lm_inp
from google3.learning.brain.research.babelfish.lm import layers as lm_layers
from google3.learning.brain.research.babelfish.lm import model2


# TODO(jmluo)
# Extend one_billion_wds_word_level instead -- dont't have to write all the stuff

class WordLevelPTBBase(base_model_params.SingleTaskModelParams):
  """Params for training a word-level LM on PTB."""

  # One Billion Words benchmark corpus is available in iq, li and ok.
  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/ptb/')
  EMBEDDING_DIM = 512
  MAX_TOKENS = 512
  NUM_EMBEDDING_SHARDS = 1
  NUM_SAMPLED = 9999
  NUM_SOFTMAX_SHARDS = 1
  RNN_STATE_DIM = 512
  VOCAB_SIZE = 10000  # includes <epsilon>, vocabulary in fst symtable format
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'lm.vocabulary.syms')

  @classmethod
  def Train(cls):
    p = lm_inp.LmInput.Params()
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    # p.bucket_batch_limit = [128] * len(p.bucket_upper_bound) # [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    p.file_buffer_size = 10000000
    p.file_parallelism = 10
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'ptb.train.sst-00000-of-00001')
    p.name = 'ptb_train_set'
    p.tokenizer = tokenizers.VocabFileTokenizer.Params()
    p.tokenizer.normalization = ''
    p.num_batcher_threads = 16
    p.target_max_length = cls.MAX_TOKENS
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 3
    p.tokenizer.token_vocab_filepath = cls.WORD_VOCAB
    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    return p

  @classmethod
  def Dev(cls):
    p = cls.Train()
    # Use small batches for eval.
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 64, 32, 32, 16, 16, 4, 2, 1]
    p.file_buffer_size = 1
    p.file_parallelism = 1
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'ptb.valid.sst-00000-of-00001')
    p.name = 'ptb_dev_set'
    p.num_batcher_threads = 1
    p.num_samples = 4900  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Test(cls):
    p = cls.Train()
    # Use small batches for eval.
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 64, 32, 32, 16, 16, 4, 2, 1]
    p.file_buffer_size = 1
    p.file_parallelism = 1
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'ptb.test.sst-00000-of-00001')
    p.name = 'ptb_test_set'
    p.num_batcher_threads = 1
    p.num_samples = 4840  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Task(cls):
    p = model2.LanguageModelV2.Params()
    p.name = 'ptb_word_level_lm'
    p.eval.samples_per_summary = 10000

    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=cls.VOCAB_SIZE,
        emb_dim=cls.EMBEDDING_DIM,
        num_layers=1,
        residual_start=3,  # disable residuals
        rnn_dims=cls.EMBEDDING_DIM,
        rnn_hidden_dims=cls.RNN_STATE_DIM)

    # Input embedding needs to be sharded.
    p.lm.emb.max_num_shards = p.lm.emb.actual_shards = cls.NUM_EMBEDDING_SHARDS
    p.lm.embedding_dropout_keep_prob = 0.5
    # Match the initialization in third_party code.
    p.lm.emb.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_EMBEDDING_SHARDS)

    # We also want dropout after each of the RNN layers.
    p.lm.rnns.dropout.keep_prob = 0.5

    # Adjusts training params.
    tp = p.train
    # Use raw loss: sum logP across tokens in a batch but average across splits.
    tp.sum_loss_across_splits = False
    tp.sum_loss_across_tokens_in_batch = True
    # Disable any so called "clipping" (gradient scaling really).
    tp.clip_gradient_norm_to_value = 0.0
    tp.grad_norm_to_clip_to_zero = 0.0
    # Do clip the LSTM gradients.
    tp.max_lstm_gradient_norm = 16
    # Straight Adagrad; very sensitive to initial accumulator value, the default
    # 0.1 value is far from adequate.
    # TODO(ciprianchelba): tune accumulator value, learning rate, clipping
    # threshold.
    tp.learning_rate = 0.1
    tp.lr_schedule = (
        lr_schedule.PiecewiseConstantLearningRateSchedule.Params().Set(
            boundaries=[], values=[1.0]))
    tp.l2_regularizer_weight = None  # No regularization.
    tp.optimizer = optimizer.Adagrad.Params()
    tp.save_interval_seconds = 100
    tp.summary_interval_steps = 20
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmax(WordLevelPTBBase):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmax, cls).Task()
    num_input_dim = p.lm.softmax.input_dim
    p.lm.softmax = layers.SimpleFullSoftmax.Params()
    p.lm.softmax.input_dim = num_input_dim
    p.lm.softmax.num_classes = cls.VOCAB_SIZE
    p.lm.softmax.num_sampled = cls.NUM_SAMPLED
    p.lm.softmax.num_shards = cls.NUM_SOFTMAX_SHARDS
    # NOTE this makes tying input and output embeddings much easier
    p.lm.emb.partition_strategy = 'div'
    # Match the initialization in third_party code.
    p.lm.softmax.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_SOFTMAX_SHARDS)
    assert p.lm.softmax.num_classes % p.lm.softmax.num_shards == 0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam(WordLevelPTBSimpleSoftmax):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam, cls).Task()
    p.train.optimizer = optimizer.Adam.Params()

    # TODO(jmluo) this is really buggy and hacky
    # use uniform initializer (-scale, scale)
    scale = 0.08
    def iter_iter(p, pattern):
      for name, param in p.IterParams():
        if hasattr(param, 'IterParams'):
          if pattern in name:
            d = {name: py_utils.WeightInit.Uniform(scale=scale)}
            p.Set(**d)
          else:
            iter_iter(param, pattern)
        elif isinstance(param, list):
          for cell_p in param:
            if hasattr(cell_p, 'IterParams'):
              cell_p.Set(params_init=py_utils.WeightInit.Uniform(scale=scale))
    iter_iter(p, 'params_init')

    # forget gate bias set to 1.0
    for param in p.lm.rnns.cell_tpl:
      param.forget_gate_bias = 1.0

    # gradient norm clipping
    p.train.clip_gradient_norm_to_value = 5.0
    p.train.grad_norm_to_clip_to_zero = 0.0
    p.train.max_lstm_gradient_norm = 0

    # Use SGD and dev-based decay learning schedule
#     p.train.lr_schedule = (
#         lr_schedule.DevBasedSchedule.Params().Set(decay=0.9))
#     p.train.optimizer = optimizer.SGD.Params()
#     p.train.learning_rate = 1.0
#
#     p.train.clip_gradient_norm_to_value = 5.0

    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam53(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam53, cls).Task()
    p.train.learning_rate = 5e-3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam23(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam23, cls).Task()
    p.train.learning_rate = 2e-3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam23Reg14(WordLevelPTBSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam23Reg14, cls).Task()
    p.train.learning_rate = 2e-3
    p.train.l2_regularizer_weight = 1e-4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam23Reg24(WordLevelPTBSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam23Reg24, cls).Task()
    p.train.learning_rate = 2e-3
    p.train.l2_regularizer_weight = 2e-4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam23Reg54(WordLevelPTBSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam23Reg54, cls).Task()
    p.train.learning_rate = 2e-3
    p.train.l2_regularizer_weight = 5e-4
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam13(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam13, cls).Task()
    p.train.learning_rate = 1e-3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam54(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam54, cls).Task()
    p.train.learning_rate = 5e-4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam24(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam24, cls).Task()
    p.train.learning_rate = 2e-4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxAdam14(WordLevelPTBSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxAdam14, cls).Task()
    p.train.learning_rate = 1e-4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRR(WordLevelPTBSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRR, cls).Task()
    old_params = p.lm.emb
    hrr = lm_layers.HRREmbeddingLayer.Params()
    hrr.s = old_params.Copy()
    hrr.e_l = old_params.Copy()
    hrr.vocab_size = hrr.e_l.vocab_size = cls.VOCAB_SIZE
    hrr.s.vocab_size = cls.VOCAB_SIZE
    hrr.embedding_dim = hrr.e_l.embedding_dim = cls.EMBEDDING_DIM
    hrr.num_roles = cls.NUM_ROLES
    hrr.num_fillers_per_role = cls.NUM_FILLERS_PER_ROLE
    hrr.s.embedding_dim = cls.NUM_FILLERS_PER_ROLE * cls.NUM_ROLES
    hrr.actual_shards = hrr.e_l.actual_shards
    p.lm.emb = hrr
    p.lm.softmax.input_dim *= cls.NUM_ROLES # size: |V| x nr*d
    # TODO(jmluo)
    # add dropout for r and F
    p.lm.decode_dropout_keep_prob = 0.5
    p.lm.softmax.num_roles = cls.NUM_ROLES
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso13(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso13, cls).Task()
    p.train.isometric = 1e-3
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso12(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso12, cls).Task()
    p.train.isometric = 1e-2
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso11(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso11, cls).Task()
    p.train.isometric = 1e-1
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso10(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso10, cls).Task()
    p.train.isometric = 1e-0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso110(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso110, cls).Task()
    p.train.isometric = 1e1
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso120(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso120, cls).Task()
    p.train.isometric = 1e2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso130(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso130, cls).Task()
    p.train.isometric = 1e3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso140(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso140, cls).Task()
    p.train.isometric = 1e4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2(WordLevelPTBSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 2
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2, cls).Task()
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR3(WordLevelPTBSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 3
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR3, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR5(WordLevelPTBSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 5
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR5, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR10(WordLevelPTBSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 10
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR10, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2Tie(WordLevelPTBSimpleSoftmaxHRRIsoR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2(WordLevelPTBSimpleSoftmaxHRRIsoR2Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2G(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2G, cls).Task()
    p.lm.softmax.gating = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50G(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2G):
  """Use sampled soft-max in training."""

  NUM_FILLERS_PER_ROLE = 50

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50G, cls).Task()
    p.lm.softmax.gating = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GD(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50G):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GD, cls).Task()
    p.lm.softmax.dropout = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50G):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GA, cls).Task()
    p.lm.softmax.anneal = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50G):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA, cls).Task()
    p.lm.softmax.role_anneal = 3000
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):

  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/ptb/chunk/')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.chunk')
  NUM_SAMPLED = 6335
  VOCAB_SIZE = 6336  # includes <epsilon>, vocabulary in fst symtable format


  @classmethod
  def Train(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50CGRA, cls).Train()
    p.gold_chunks = True
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'train.chunk.sst-00000-of-00001')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [64] * len(p.bucket_upper_bound)
    return p

  @classmethod
  def Dev(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50CGRA, cls).Dev()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'dev.chunk.sst-00000-of-00001')
    return p

  @classmethod
  def Test(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50CGRA, cls).Test()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'test.chunk.sst-00000-of-00001')
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA, cls).Task()
    p.lm.emb.mode = 'rs'
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoLR13R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoLR13R2TieNR2NF50RSGRA, cls).Task()
    p.train.learning_rate = 1e-3
    return p



@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoLR54R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoLR54R2TieNR2NF50RSGRA, cls).Task()
    p.train.learning_rate = 5e-4
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso130R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso130R2TieNR2NF50RSGRA, cls).Task()
    p.train.isometric = 1e3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso120R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso120R2TieNR2NF50RSGRA, cls).Task()
    p.train.isometric = 1e2
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso110R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso110R2TieNR2NF50RSGRA, cls).Task()
    p.train.isometric = 1e1
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIso10R2TieNR2NF50RSGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50RSGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIso10R2TieNR2NF50RSGRA, cls).Task()
    p.train.isometric = 1e0
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50DOGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50DOGRA, cls).Task()
    p.lm.emb.mode = 'dec_only'
    return p



@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50DO1000GRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50DOGRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50DO1000GRA, cls).Task()
    p.lm.softmax.role_anneal = 1000
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10GRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10GRA, cls).Task()
    p.lm.num_sent_roles = 10
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA, cls).Task()
    p.lm.num_sent_roles = 3
    p.lm.global_decode = True
    p.train.global_decode = 1e4
    for tpl in p.lm.rnns.cell_tpl:
      tpl.num_output_nodes = 2 * cls.EMBEDDING_DIM
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalChunkGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalChunkGRA, cls).Task()
    p.lm.chunk_loss = True
    for tpl in p.lm.rnns.cell_tpl:
      tpl.num_output_nodes = 3 * cls.EMBEDDING_DIM
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalChunkGRA):

  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/ptb/chunk/')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.chunk')
  NUM_SAMPLED = 6335
  VOCAB_SIZE = 6336  # includes <epsilon>, vocabulary in fst symtable format


  @classmethod
  def Train(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Train()
    p.gold_chunks = True
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'train.chunk.sst-00000-of-00001')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [64] * len(p.bucket_upper_bound)
    return p

  @classmethod
  def Dev(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Dev()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'dev.chunk.sst-00000-of-00001')
    return p

  @classmethod
  def Test(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Test()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'test.chunk.sst-00000-of-00001')
    return p

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Task()
    # TODO(jmluo) need to rename this -- I'm still using chunk loss but there is no r_o prediction.
    p.lm.chunk_loss = False
    p.lm.gold_chunks = True
    p.lm.cc_size = 10 #cls.VOCAB_SIZE
    for tpl in p.lm.rnns.cell_tpl:
      tpl.num_output_nodes = 2 * cls.EMBEDDING_DIM
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA, cls).Task()
    p.lm.num_sent_roles = 2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z150CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z150CGRA, cls).Task()
    p.train.zipf = 100000.0
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA, cls).Task()
    p.train.zipf = 10000.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC10CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC10CGRA, cls).Task()
    p.train.cc_entropy = 1.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC10S130CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC10CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC10S130CGRA, cls).Task()
    p.lm.cc_size = 1000
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA, cls).Task()
    p.train.cc_entropy = 10.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110SCGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110SCGRA, cls).Task()
    p.lm.chunk_input_type = 'sent_act'
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z130CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z130CGRA, cls).Task()
    p.train.zipf = 1000.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z120CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z120CGRA, cls).Task()
    p.train.zipf = 100.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z110CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z110CGRA, cls).Task()
    p.train.zipf = 10.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z10CGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z10CGRA, cls).Task()
    p.train.zipf = 1.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GD10GCGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalChunkGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GD10GCGRA, cls).Task()
    p.train.global_decode = 1e0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10GlobalChunkGRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalChunkGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10GlobalChunkGRA, cls).Task()
    p.lm.num_sent_roles = 10
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10V140GRA(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50SR10V140GRA, cls).Task()
    p.train.variance = 1e4
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D10(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D10, cls).Task()
    p.train.diverse = 1.0
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D110(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D110, cls).Task()
    p.train.diverse = 1e1
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D120(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D120, cls).Task()
    p.train.diverse = 1e2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D130(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D130, cls).Task()
    p.train.diverse = 1e3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD10(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD10, cls).Task()
    p.train.kld = 1.0
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD110(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD110, cls).Task()
    p.train.kld = 10.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD120(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD120, cls).Task()
    p.train.kld = 100.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD130(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD130, cls).Task()
    p.train.kld = 1000.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50D130(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2D130):
  """Use sampled soft-max in training."""

  NUM_FILLERS_PER_ROLE = 50

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2NF50D130, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD140(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2):

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR2TieNR2KLD140, cls).Task()
    p.train.kld = 10000.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR3Tie(WordLevelPTBSimpleSoftmaxHRRIsoR3):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR3Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR3TieNR3(WordLevelPTBSimpleSoftmaxHRRIsoR3Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR3TieNR3, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR5Tie(WordLevelPTBSimpleSoftmaxHRRIsoR5):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR5Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR5TieNR5(WordLevelPTBSimpleSoftmaxHRRIsoR5Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR5TieNR5, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRIsoR10Tie(WordLevelPTBSimpleSoftmaxHRRIsoR10):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRIsoR10Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxTie(WordLevelPTBSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxTie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxTieC(WordLevelPTBSimpleSoftmaxTie):

  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/ptb/chunk/')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.chunk')
  NUM_SAMPLED = 6335
  VOCAB_SIZE = 6336  # includes <epsilon>, vocabulary in fst symtable format

  @classmethod
  def Train(cls):
    p = super(WordLevelPTBSimpleSoftmaxTieC, cls).Train()
    p.gold_chunks = True
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'train.chunk.sst-00000-of-00001')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [64] * len(p.bucket_upper_bound)
    return p

  @classmethod
  def Dev(cls):
    p = super(WordLevelPTBSimpleSoftmaxTieC, cls).Dev()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'dev.chunk.sst-00000-of-00001')
    return p

  @classmethod
  def Test(cls):
    p = super(WordLevelPTBSimpleSoftmaxTieC, cls).Test()
    p.num_samples = 1006
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, 'test.chunk.sst-00000-of-00001')
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxTieTC(WordLevelPTBSimpleSoftmaxTie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxTieTC, cls).Task()
    p.lm.two_copies = True
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg13(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg13, cls).Task()
    p.train.l1_reg_weight = 1e-3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg12(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg12, cls).Task()
    p.train.l1_reg_weight = 1e-2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg11(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg11, cls).Task()
    p.train.l1_reg_weight = 1e-1
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg10(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg10, cls).Task()
    p.train.l1_reg_weight = 1e-0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg110(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg110, cls).Task()
    p.train.l1_reg_weight = 1e1
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg120(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg120, cls).Task()
    p.train.l1_reg_weight = 1e2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg130(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg130, cls).Task()
    p.train.l1_reg_weight = 1e3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevelPTBSimpleSoftmaxHRRL1Reg140(WordLevelPTBSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevelPTBSimpleSoftmaxHRRL1Reg140, cls).Task()
    p.train.l1_reg_weight = 1e4
    return p

