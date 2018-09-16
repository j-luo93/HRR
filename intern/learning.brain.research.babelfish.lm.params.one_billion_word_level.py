"""Train word-level LMs on 1 billion word data."""

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

class WordLevel1BBase(base_model_params.SingleTaskModelParams):
  """Params for training a word-level LM on 1B."""

  # One Billion Words benchmark corpus is available in iq, li and ok.
  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/1b/')
  EMBEDDING_DIM = 512
  MAX_TOKENS = 512
  NUM_EMBEDDING_SHARDS = 8
  NUM_SAMPLED = 4096
  NUM_SOFTMAX_SHARDS = 8
  RNN_STATE_DIM = 512
  # VOCAB_SIZE = 218159  # includes <epsilon>, vocabulary in fst symtable format
  VOCAB_SIZE = 218160 # make sure it's a multiple of 8
  WORD_VOCAB = os.path.join(CORPUS_DIR, '1b.3M.vocabulary.syms')

  @classmethod
  def Train(cls):
    p = lm_inp.LmInput.Params()
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    p.file_buffer_size = 10000000
    p.file_parallelism = 10
    p.file_pattern = os.path.join(
        cls.CORPUS_DIR, '1b.train.3M.sst-00000-of-00001')
    p.name = '1b_train_set'
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
        cls.CORPUS_DIR, '1b.dev.3M.sst-00000-of-00001')
    p.name = '1b_dev_set'
    p.num_batcher_threads = 1
    p.num_samples = 6206  # Number of sentences to evaluate on.
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
        cls.CORPUS_DIR, '1b.test.3M.sst-00000-of-00001')
    p.name = '1b_test_set'
    p.num_batcher_threads = 1
    p.num_samples = 6075  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Task(cls):
    p = model2.LanguageModelV2.Params()
    p.name = '1b_word_level_lm'
    p.eval.samples_per_summary = 10000

    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=cls.VOCAB_SIZE,
        emb_dim=cls.EMBEDDING_DIM,
        num_layers=2,
        residual_start=3,  # disable residuals
        rnn_dims=cls.EMBEDDING_DIM,
        rnn_hidden_dims=cls.RNN_STATE_DIM)

    # Input embedding needs to be sharded.
    p.lm.emb.max_num_shards = p.lm.emb.actual_shards = cls.NUM_EMBEDDING_SHARDS
    p.lm.embedding_dropout_keep_prob = 0.75
    # Match the initialization in third_party code.
    p.lm.emb.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_EMBEDDING_SHARDS)

    # We also want dropout after each of the RNN layers.
    p.lm.rnns.dropout.keep_prob = 0.75

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
    tp.save_interval_seconds = 200
    tp.summary_interval_steps = 100
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmax(WordLevel1BBase):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmax, cls).Task()
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
class WordLevel1BSimpleSoftmaxAdam(WordLevel1BSimpleSoftmax):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxAdam, cls).Task()
    p.train.optimizer = optimizer.Adam.Params()

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
class WordLevel1BSimpleSoftmaxAdam23(WordLevel1BSimpleSoftmaxAdam):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxAdam23, cls).Task()
    p.train.learning_rate = 2e-3
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRR(WordLevel1BSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRR, cls).Task()
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
    p.lm.decode_dropout_keep_prob = 0.75
    p.lm.softmax.num_roles = cls.NUM_ROLES
    return p



@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIso140(WordLevel1BSimpleSoftmaxHRR):
  """Use sampled soft-max in training."""

  NUM_ROLES = 1
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIso140, cls).Task()
    p.train.isometric = 1e4
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2(WordLevel1BSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 2
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2, cls).Task()
    return p


@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR3(WordLevel1BSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 3
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR3, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR4(WordLevel1BSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 4
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR4, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR5(WordLevel1BSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 5
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR5, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR10(WordLevel1BSimpleSoftmaxHRRIso140):
  """Use sampled soft-max in training."""

  NUM_ROLES = 10
  NUM_FILLERS_PER_ROLE = 20

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR10, cls).Task()
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2Tie(WordLevel1BSimpleSoftmaxHRRIsoR2):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p


@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2(WordLevel1BSimpleSoftmaxHRRIsoR2Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p


@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2):
  """Use sampled soft-max in training."""
  NUM_FILLERS_PER_ROLE = 50

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50GRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50GRA, cls).Task()
    p.lm.softmax.role_anneal = 3000
    p.lm.softmax.gating = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR3Tie(WordLevel1BSimpleSoftmaxHRRIsoR3):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR3Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR3TieNR3(WordLevel1BSimpleSoftmaxHRRIsoR3Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR3TieNR3, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR4Tie(WordLevel1BSimpleSoftmaxHRRIsoR4):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR4Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR4TieNR4(WordLevel1BSimpleSoftmaxHRRIsoR4Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR4TieNR4, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR5Tie(WordLevel1BSimpleSoftmaxHRRIsoR5):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR5Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR5TieNR5(WordLevel1BSimpleSoftmaxHRRIsoR5Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR5TieNR5, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR10Tie(WordLevel1BSimpleSoftmaxHRRIsoR10):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR10Tie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR10TieNR10(WordLevel1BSimpleSoftmaxHRRIsoR10Tie):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR10TieNR10, cls).Task()
    p.lm.num_roles = cls.NUM_ROLES
    return p


@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxTie(WordLevel1BSimpleSoftmaxAdam23):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxTie, cls).Task()
    p.lm.tie = True
    p.lm.softmax.tie = True
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50GRA):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA, cls).Task()
    p.lm.num_sent_roles = 3
    p.lm.global_decode = True
    p.train.global_decode = 1e4
    tpl = p.lm.rnns.cell_tpl[-1]
    tpl.num_output_nodes = 2 * cls.EMBEDDING_DIM
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3GlobalGRA):

  CORPUS_DIR = os.path.join('/cns/jn-d/home/jmluo/brain/rs=6.3/',
                            'data/1b/3M/chunk/')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'lm.vocab.chunk')
  VOCAB_SIZE = 218168


  @classmethod
  def Train(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Train()
    p.gold_chunks = True
    return p

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA, cls).Task()
    # TODO(jmluo) need to rename this -- I'm still using chunk loss but there is no r_o prediction.
    p.lm.chunk_loss = False
    p.lm.gold_chunks = True
    p.lm.cc_size = 10 #cls.VOCAB_SIZE
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR3CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA, cls).Task()
    p.lm.num_sent_roles = 2
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA, cls).Task()
    p.train.zipf = 10000.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110CGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110CGRA, cls).Task()
    p.train.cc_entropy = 10.0
    return p

@model_registry.RegisterSingleTaskModel
class WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110SCGRA(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110CGRA):

  @classmethod
  def Task(cls):
    p = super(WordLevel1BSimpleSoftmaxHRRIsoR2TieNR2NF50SR2Z140CC110SCGRA, cls).Task()
    p.lm.chunk_input_type = 'sent_act'
    return p

