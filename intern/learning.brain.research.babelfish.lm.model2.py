"""LM models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from google3.learning.brain.research.babelfish import py_utils
from google3.third_party.tensorflow_lingvo.core import lr_schedule
from google3.learning.brain.research.babelfish.lm import input_generator as lm_inp
from google3.third_party.tensorflow_lingvo.tasks.lm import input_generator as lingvo_lm_inp
from google3.third_party.tensorflow_lingvo.tasks.lm import model as lingvo_model
from google3.learning.brain.research.babelfish.lm import layers


class LanguageModelV2(lingvo_model.LanguageModel):
  """Language model version 2."""

  @classmethod
  def Params(cls):
    # Have to use the Params() method from the grandparent -- to skip initializing lm twice.
    p = super(lingvo_model.LanguageModel, cls).Params()
    p.Define('lm', layers.RnnLm.Params(), 'LM layer.')
    tp = p.train
    tp.Define(
        'max_lstm_gradient_norm', 0.0,
        'Clip gradient for vars in lstm layers by setting this value to '
        'something > 0.')
    tp.Define('sum_loss_across_splits', False,
          'Sum batch-level loss across splits/towers when set to True; '
          'average across splits/towers weighted by the ratio number '
          'predicted tokens in split / total number of predicted tokens '
          'across splits when set to False (default).')
    tp.Define(
        'sum_loss_across_tokens_in_batch', False,
        'Sum the logP across predicted tokens in batch when set to True; '
        'average across predicted tokens in batch o/w (default).')
    tp.Define('l1_reg_weight', 0.0,
              'L1 regularization weight for signature distributions')
    tp.Define('isometric', 0.0, 'Weight for isometric constraint')
    tp.Define('kld', 0.0, 'Weight for KLD')
    tp.Define('diverse', 0.0, 'Weight for hubness penalty') # TODO(jmluo) rename this
    tp.Define('variance', 0.0, 'Weight for sentence role variance')
    tp.Define('global_decode', 0.0, 'Weight for global decoding')
    tp.Define('zipf', 0.0, 'Weight for KLD wrt zipf')
    tp.Define('cc_entropy', 0.0, 'Weight for KLD wrt zipf')
    tp.lr_schedule = lr_schedule.PiecewiseConstantLearningRateSchedule.Params(
    ).Set(
        boundaries=[350000, 500000, 600000], values=[1.0, 0.1, 0.01, 0.001])
    tp.vn_start_step = 20000
    tp.vn_std = 0.0
    tp.learning_rate = 0.001
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    return p

  def _TrimIfPossibleThenTranspose(self, ids, paddings, labels, weights, chunk_ids=None):
    data = (ids, paddings, labels, weights)
    if not py_utils.use_tpu():
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)
      data = (x[:, :max_seq_len] for x in data)
      if chunk_ids is not None:
        # chunk_ids, last_word_marks = chunk_ids
        chunk_ids = tf.transpose(chunk_ids[:, :max_seq_len])
        # last_word_marks = tf.transpose(last_word_marks[:, :max_seq_len])
    return [tf.transpose(x) for x in data] + [chunk_ids] #[(chunk_ids, last_word_marks)]

  def _GetInput(self, input_batch):
    p = self.params
    chunk_ids = None
    if p.input.gold_chunks:
      chunk_ids = input_batch.chunk_ids
    return self._TrimIfPossibleThenTranspose(
        input_batch.ids, input_batch.paddings, input_batch.labels,
        input_batch.weights, chunk_ids=chunk_ids)

  def FPropTower(self, theta, input_batch):
    p = self.params
    if p.input and p.input.cls not in (lm_inp.LmInput,
                                       lm_inp.FixedSizeRandomLmInput):
      input_batch = lm_inp.SpeechInputAdapter(input_batch)
    # chunk_ids is None if no chunk tags are used
    ids, paddings, labels_ids, weights, chunk_ids = self._GetInput(input_batch)

    seqlen = tf.shape(ids)[0]
    batch_size = tf.shape(ids)[1]
    state0 = self.lm.zero_state(batch_size)
    labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)

    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      lower_state0 = self.lm.lower_rnns.zero_state(batch_size)
      xent_output, _, _ = self.lm.FProp(theta.lm, ids, paddings, state0,
                                     labels=labels, lower_state0=lower_state0, chunk_ids=chunk_ids)
    else:
      xent_output, _ = self.lm.FProp(theta.lm, ids, paddings, state0, labels=labels, chunk_ids=chunk_ids)



    # +1 to account for the end of sequence symbol.
    div = 2 if p.input.gold_chunks else 1 # tags shouldn't be counted as words
    num_words = tf.cast(
        tf.reduce_sum(input_batch.word_count // div + tf.constant(1, dtype=tf.int32)),
        tf.float32)
    predicted_labels = tf.cast(xent_output.per_example_argmax, labels_ids.dtype)

    num_preds = xent_output.total_weight
    mean_acc = tf.reduce_sum(
        tf.cast(tf.equal(labels_ids, predicted_labels), tf.float32) * weights
    ) / (
        num_preds + 1e-4)

    if p.lm.emb.cls == layers.HRREmbeddingLayer:
      signature = xent_output.signature
      if p.train.l1_reg_weight > 0.0:
        _, _, ns, nf = signature.get_shape().as_list()
        l1 = tf.reduce_sum(tf.abs(signature)) / tf.cast(batch_size * ns * nf, tf.float32)
        l1_loss = p.train.l1_reg_weight * l1
      if p.train.isometric > 0.0:
        isometric_constraint = 0.0
        nr = p.lm.emb.num_roles
        # TODO(jmluo) rearrange it to divide the code according to three modes
        if 'F' in theta.lm.emb:
          F_wm = theta.lm.emb.F
          nr, nf, d = F_wm.get_shape().as_list()
          # F2d leads to overspefication of parameters in F
          F2d = tf.reshape(F_wm, [nr * nf, d])
          diff = tf.matmul(F2d, tf.transpose(F2d)) - tf.eye(nr * nf)
          # diff = tf.matmul(F_wm, tf.transpose(F_wm, perm=[0, 2, 1])) - tf.eye(nf)
          isometric_constraint += tf.reduce_sum(diff**2) / 2.0
        if 'A' in theta.lm:
          d = theta.lm.A.get_shape().as_list()[0]
          A = tf.reshape(theta.lm.A, [d, 2, d])
          A1 = A[:, 0]
          A2 = A[:, 1]
          diff = tf.matmul(A1, tf.transpose(A2)) / 2
          # isometric_constraint += tf.reduce_sum(diff ** 2)

        if nr > 1 and 'r' in theta.lm.emb:
          r_wm = theta.lm.emb.r
          diff = tf.matmul(r_wm, tf.transpose(r_wm)) - tf.eye(nr)
          isometric_constraint += tf.reduce_sum(diff**2)
        if 'R' in theta.lm:
          R_wm = theta.lm.R
          diff = tf.matmul(R_wm, tf.transpose(R_wm)) - tf.eye(p.lm.num_sent_roles)
          isometric_constraint += tf.reduce_sum(diff**2)
        if p.lm.emb.mode == 'rs':
          assert 'rR' in theta.lm.emb
          rR = theta.lm.emb.rR
          diff = tf.matmul(rR, tf.transpose(rR)) - tf.eye(2)
          isometric_constraint += tf.reduce_sum(diff ** 2)

          rs_all = theta.lm.emb.rs.wm
          for rs in rs_all:
            rs = tf.reshape(rs, [-1, 2, 2])
            # # old code
            # # rs = tf.nn.l2_normalize(rs, axis=-1)
            # cosine = tf.matmul(rs, tf.transpose(rs, perm=[0, 2, 1]))
            # # diff = tf.abs(cosine * (tf.ones([2, 2]) - tf.eye(2))) # only take the off-diagonal entries
            # diff = cosine - tf.eye(2)
            # isometric_constraint += tf.reduce_sum(diff ** 2)

            # one-hot loss from TPR
            norm = tf.reduce_sum(rs ** 2, axis=-1)
            isometric_constraint += tf.reduce_sum((norm - 1.0) ** 2) + tf.reduce_sum((rs ** 2) * ((1 - rs) ** 2))

            normalized_rs = tf.nn.l2_normalize(rs, axis=-1)
            dot = tf.matmul(normalized_rs, tf.transpose(normalized_rs, perm=[0, 2, 1]))
            isometric_constraint += tf.reduce_sum(((dot * (tf.ones([2, 2]) - tf.eye(2))) ** 2) * 0.5)
          tf.summary.histogram('rs', tf.stack(rs_all))
        isometric_loss = isometric_constraint * p.train.isometric

    if p.train.kld > 0.0 and not p.is_eval:
      assert xent_output.all_logits is not None
      assert p.lm.num_roles == 2

      def compute_KLD(p0, p1):
        with tf.name_scope('KLD'):
          log_diff = tf.log(p1 + 1e-8) - tf.log(p0 + 1e-8)
          kld_loss = p0 * log_diff # no minus sign here, because we want to increase KLD
          return p.train.kld * tf.reduce_sum(kld_loss) / num_preds

      all_probs = [tf.nn.softmax(logits) for logits in xent_output.all_logits]
      p0, p1 = all_probs
      kld_loss = compute_KLD(p0, p1) + compute_KLD(p1, p0) # symmetrized

    if p.train.diverse > 0.0 and not p.is_eval:
      assert xent_output.all_logits is not None
      assert p.lm.num_roles == 2

      values = list()
      indices = list()
      for logits in xent_output.all_logits:
        val, ind = tf.nn.top_k(logits, k=10, sorted=False)
        values.append(val)
        indices.append(ind)
      int_ind = tf.sets.set_intersection(*indices)
      bs = tf.shape(ids)[0]
      sl = tf.shape(ids)[1]
      K = tf.shape(int_ind)[-1]
      int_ind = tf.reshape(tf.sparse_tensor_to_dense(int_ind), [bs, sl, K])
      bs_ind = tf.expand_dims(tf.expand_dims(tf.range(bs), axis=-1), axis=-1)
      bs_ind = tf.tile(bs_ind, [1, sl, K])
      sl_ind = tf.expand_dims(tf.expand_dims(tf.range(sl), axis=0), axis=-1)
      sl_ind = tf.tile(sl_ind, [bs, 1, K])
      gather_ind = tf.stack([bs_ind, sl_ind, int_ind], axis=-1)
      logits = tf.reshape(xent_output.all_logits[-1], [bs, sl, -1])
      hubness_loss = tf.reduce_sum(tf.gather_nd(logits, gather_ind) * tf.cast(int_ind > 0, tf.float32)) / num_preds * p.train.diverse
      # scale_factor = tf.cast(int_ind > 0, tf.float32)
      # hubness_loss = p.train.diverse * scale_factor * tf.reduce_sum(values[-1]) / 10

    if p.train.variance > 0.0 and not p.is_eval:
      assert p.lm.num_sent_roles > 0
      lr = xent_output.lower_roles
      hr = xent_output.higher_roles
      lr_mean, lr_var = tf.nn.moments(lr, axes=[0])
      hr_mean, hr_var = tf.nn.moments(hr, axes=[0])
      lr_var = py_utils.HasRank(lr_var, 1)
      hr_var = py_utils.HasRank(hr_var, 1)
      variance_loss = -p.train.variance * (tf.reduce_mean(lr_var) + tf.reduce_mean(hr_var))

    if p.train.global_decode > 0.0 and not p.is_eval:
      with tf.name_scope('global_decode'):
        assert p.lm.num_sent_roles > 0
        assert p.lm.global_decode
        role_probs = xent_output.lower_roles
        py_utils.HasRank(weights, 2)
        emb = xent_output.emb * tf.expand_dims(weights, axis=-1)
        roles = py_utils.Matmul(role_probs, theta.lm.R)
        roles = tf.reshape(roles, [seqlen, batch_size, -1])
        bound = layers.HRREmbeddingLayer.static_circular_conv(roles, emb) # size: sl x bs x d
        if p.lm.chunk_loss:
#           pred_role_probs = xent_output.pred_roles
#           pred_roles = py_utils.Matmul(pred_role_probs, theta.lm.R)
#           pred_roles = tf.reshape(pred_roles, [seqlen, batch_size, -1])
#           forward_sentence_emb = tf.cumsum(bound, reverse=True, exclusive=True, axis=0)
#           target_chunks = layers.HRREmbeddingLayer.static_decode(forward_sentence_emb, pred_roles) # size: sl x bs x d
#           normalized_target_chunks = tf.nn.l2_normalize(target_chunks, axis=-1)
#           normalized_chunk = tf.nn.l2_normalize(xent_output.predicted_chunk, axis=-1)
#           total_chunk_loss = -tf.reduce_sum(normalized_target_chunks * normalized_chunk)
#           chunk_loss = total_chunk_loss / num_preds
#
#           normalized_emb = tf.nn.l2_normalize(emb, axis=-1)
#           cosine = tf.reduce_sum(tf.expand_dims(normalized_emb, axis=1) * normalized_chunk, axis=-1) # sl x sl x bs
#           max_cosine = tf.reduce_max(cosine, axis=0) # sl x bs
#           global_coherence = -tf.reduce_mean(max_cosine) * p.train.global_decode

          # negative sampling (kinda)
          assert p.lm.num_sent_roles == 3
          pred_role_probs = xent_output.pred_roles
          _, var = tf.nn.moments(pred_role_probs, axes=[0])
          tf.summary.scalar('var0_pred_role_probs', var[0])
          tf.summary.scalar('var1_pred_role_probs', var[1])
          tf.summary.scalar('var2_pred_role_probs', var[2])
          # pred_roles = tf.matmul(pred_role_probs, theta.lm.R)
          # pred_roles = tf.reshape(pred_roles, [seqlen, batch_size, -1])
          backward_sentence_emb = tf.cumsum(bound, reverse=True, exclusive=True, axis=0)
          forward_sentence_emb = tf.cumsum(bound, axis=0)
          # chunks from ro premutation
          all_perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
          all_role_probs = tf.gather(pred_role_probs, all_perms, axis=-1)
          py_utils.HasRank(all_role_probs, 3)
          all_role_probs = tf.reshape(all_role_probs, [-1, 3])
          all_roles = tf.matmul(all_role_probs, theta.lm.R)
          all_roles = tf.reshape(all_roles, [seqlen, batch_size, 6, -1])
          all_chunks = layers.HRREmbeddingLayer.static_decode(tf.expand_dims(backward_sentence_emb, axis=-2), all_roles) # size: sl x bs x 6 x d
          # chunks from ri pertubation
          noisy_bound = layers.HRREmbeddingLayer.static_circular_conv(theta.lm.R, tf.expand_dims(emb, axis=2))
          noisy_backward_sentence_emb = tf.cumsum(noisy_bound, reverse=True, exclusive=True, axis=0)
          noisy_all_chunks = layers.HRREmbeddingLayer.static_decode(noisy_backward_sentence_emb, tf.expand_dims(roles, axis=2))
          all_chunks = tf.concat([all_chunks, noisy_all_chunks], axis=2)
          # current chunk
          current_chunk = layers.HRREmbeddingLayer.static_decode(forward_sentence_emb, roles) # size: sl x bs x d
          pred_chunk = tf.matmul(tf.reshape(current_chunk, [seqlen * batch_size, -1]), theta.lm.pred)
          pred_chunk = tf.reshape(pred_chunk, [seqlen, batch_size, -1])
          dot_prod = tf.reduce_sum(tf.expand_dims(pred_chunk, axis=-2) * all_chunks, axis=-1) # size: sl x bs x 6
          chunk_log_probs = tf.nn.log_softmax(dot_prod, axis=-1)
          total_chunk_loss = -tf.reduce_sum(weights * chunk_log_probs[..., 0])
          # anneal
          global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
          temperature = tf.minimum(tf.constant(p.lm.softmax.role_anneal * 1.0), global_step) / (p.lm.softmax.role_anneal * 1.0)
          tf.summary.scalar('chunk/temperature', temperature)
          total_chunk_loss *= temperature
          chunk_loss = total_chunk_loss / num_preds
          # global coherence
          normalized_emb = tf.nn.l2_normalize(emb, axis=-1)
          normalized_chunk = tf.nn.l2_normalize(all_chunks[..., 0, :], axis=-1)
          cosine = tf.reduce_sum(tf.expand_dims(normalized_emb, axis=1) * normalized_chunk, axis=-1) # sl x sl x bs
          max_cosine = tf.reduce_max(cosine, axis=0) # sl x bs
          global_coherence = -tf.reduce_mean(max_cosine * weights) * p.train.global_decode * temperature
        elif p.lm.gold_chunks:
          total_chunk_loss = -tf.reduce_sum(xent_output.chunk_log_probs)
          global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
          temperature = tf.minimum(tf.constant(50000.0), global_step) / 50000.0
          tf.summary.scalar('chunk/temperature', temperature)
          total_chunk_loss *= temperature
          chunk_loss = total_chunk_loss / xent_output.num_chunks
          avg_chunk_loss = chunk_loss
        else:
          sent_encoding = tf.reduce_sum(bound, axis=0) # size: bs x d
          chunks = layers.HRREmbeddingLayer.static_decode(tf.expand_dims(sent_encoding, axis=1), theta.lm.R) # size: bs x nr x d
          normalized_chunks = tf.nn.l2_normalize(chunks, axis=-1)
          # global coherence loss
          normalized_emb = tf.nn.l2_normalize(emb, axis=-1)
          cosine = tf.reduce_sum(tf.expand_dims(normalized_emb, axis=2) * normalized_chunks, axis=-1) # sl x bs x nr
          max_cosine = tf.reduce_max(cosine, axis=0) # bs x nr
          global_coherence = -tf.reduce_mean(max_cosine) * p.train.global_decode
          # global variance loss
          chunk_sim = tf.matmul(normalized_chunks, tf.transpose(normalized_chunks, perm=[0, 2, 1])) # size: bs x nr x nr; Note that diagonal entries are always 1.0 due to normalization
          global_variance = tf.reduce_mean(chunk_sim) * p.train.global_decode

    if p.train.zipf > 0.0 and not p.is_eval and False:
      with tf.name_scope('zipf_loss'):
        cc_probs = xent_output.cc_probs
        cc_entropy = tf.reduce_sum((cc_probs * (1.0 - cc_probs)) ** 2) / tf.to_float(batch_size) * p.train.cc_entropy * 0.0
        # cc_entropy = tf.reduce_sum(-cc_probs * tf.log(1e-8 + cc_probs)) / xent_output.num_chunks * p.train.cc_entropy * 100.0
        tf.summary.histogram('cc_probs', cc_probs)
        cc_cnt = tf.reduce_sum(tf.reduce_sum(cc_probs, axis=0), axis=0) # size: cc
        Z = tf.reduce_sum(cc_cnt)
        r = tf.to_float(tf.range(tf.shape(cc_cnt)[0]) + 1)
        log_cr = tf.log(cc_cnt) + tf.log(r)
        zipf_loss = tf.reduce_sum(cc_cnt * log_cr) / Z - tf.log(Z) + tf.log(tf.reduce_sum(1.0 / r))
        zipf_loss = p.train.zipf * zipf_loss #* 0.0
    # TODO(ciprianchelba):
    #
    # Raw or average loss per batch and/or across splits:
    # . default is:
    #   sum_loss_across_splits == sum_loss_across_tokens_in_batch == False
    #   corresponding to averaging across entries in each batch and then a
    # weighted average across splits/towers where each split is weighted by
    # the number of tokens in that split relative to all tokens across splits.
    # . sum_loss_across_splits == True, sum_loss_across_tokens_in_batch == True
    # did a little better than
    # sum_loss_across_splits == False, sum_loss_across_tokens_in_batch == True
    # and both did better than default in a small exp on the 1bwds corpus.
    # . the fourth option:
    # sum_loss_across_splits == True, sum_loss_across_tokens_in_batch == False
    # has not been tested in any way.
    #
    # All options are meaningful, but we plan on having just one eventually.
    split_count = num_preds
    if p.train.sum_loss_across_splits:
      split_count = tf.ones_like(num_preds)
    loss = xent_output.avg_xent
    if p.train.sum_loss_across_tokens_in_batch:
      loss = xent_output.total_xent
      if 'chunk_loss' in locals():
        chunk_loss = total_chunk_loss

    metrics = {
        'fraction_of_correct_next_step_preds': (mean_acc, num_preds),
        'log_pplx': (xent_output.avg_xent, num_preds),
        'log_pplx_per_word': (xent_output.total_xent / num_words, num_words),
        'num_predictions': (num_preds, 1),
        'num_words': (num_words, 1)
    }
    tmp_loss = loss# + theta.dummy * theta.dummy
    if 'l1_loss' in locals():
      tmp_loss += l1_loss
      metrics['l1'] = (l1, split_count)
    if 'isometric_loss' in locals():
      tmp_loss += isometric_loss
      metrics['isometric'] = (isometric_loss, 1)
    if 'kld_loss' in locals():
      tmp_loss += kld_loss
      metrics['kld_loss'] = (kld_loss, 1)
    if 'hubness_loss' in locals():
      tmp_loss += hubness_loss
      metrics['hubness_loss'] = (hubness_loss, 1)
    if 'variance_loss' in locals():
      tmp_loss += variance_loss
      metrics['variance_loss'] = (variance_loss, 1)
    if 'global_coherence' in locals():
      tmp_loss += global_coherence
      metrics['global_coherence'] = (global_coherence, 1)
    if 'global_variance' in locals():
      tmp_loss += global_variance
      metrics['global_variance'] = (global_variance, 1)
    if 'chunk_loss' in locals():
      tmp_loss += chunk_loss
      metrics['chunk_loss'] = (chunk_loss, split_count)
      metrics['avg_chunk_loss'] = (avg_chunk_loss, xent_output.num_chunks)
      metrics['num_chunks'] = (xent_output.num_chunks, 1)
    if 'zipf_loss' in locals():
      tmp_loss += zipf_loss
      metrics['zipf_loss'] = (zipf_loss, 1)
      tmp_loss += cc_entropy
      metrics['cc_entropy'] = (cc_entropy, 1)
    # if 'rs_mean' in locals():
    #   metrics['rs_mean'] = (rs_mean, 1)
    metrics['loss'] = (tmp_loss, split_count)

    return metrics

  def FProp(self, theta):
    p = self.params.train
    metrics = super(LanguageModelV2, self).FProp(theta)
    if p.sum_loss_across_splits:
      total_xent_per_batch, num_batches = metrics['loss']
      self._loss = total_xent_per_batch * num_batches
      # Doesn't matter, _num_predicts is no longer used anywhere.
      self._num_predicts = 1
      self._loss = py_utils.CheckNumerics(self._loss)
    return metrics

  def AdjustGradients(self, var_grad):
    """Clip LSTM gradients.

    Args:
      var_grad: a NestedMap of (variable, gradient). You can view
      var_grad as an ordered list of (key, (var, grad)) tuples. Every
      key of var_grad exists in vmap. Every variable in vmap that
      contributes to loss must exist in var_grad. Every var of var_grad
      must exist in vmap.  grad is the corresponding gradient computed
      for var. grad is guaranteed to be not None.

    Returns:
      adjusted version of var_grad that has clipped the LSTM gradients
      if self.params.max_lstm_gradient_norm is set.
    """

    p = self.params
    if p.train.max_lstm_gradient_norm:
      lstm_var_grad = var_grad.lm.rnns
      lstm_vars = lstm_var_grad.Transform(lambda x: x[0]).Flatten()
      lstm_grads = lstm_var_grad.Transform(lambda x: x[1]).Flatten()
      clipped_lstm_grads, _ = tf.clip_by_global_norm(
          lstm_grads, p.train.max_lstm_gradient_norm)
      var_grad.lm.rnns = var_grad.lm.rnns.Pack(
          list(zip(lstm_vars, clipped_lstm_grads)))

    return var_grad

  def Inference(self):
    """Constructs the inference subgraphs.

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
      subgraphs['rnn_step'] = self._InferenceSubgraph_RNNStep()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph.

    Returns:
      fetches: A dictionary of fetches, containing:
        log_pplx_per_token: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's log prob.
        paddings: A matrix of shape [batch, time]. The padding mask.
        log_pplx_per_sample: A vector of shape [batch]. [i]
          is i-th input text's log prob.
        num_oovs_per_sample: A vector of shape [batch] counting the total number
          of out-of-vocabulary tokens in each input.
        tokens_from_labels: A vector of shape [batch] returning the predicted
          tokens as a sequence after mapping them back to strings from ids using
          the vocabulary.
        ids: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's id.
      feeds: A dictionary of feeds, containing:
        text: A placeholder for a vector of strings.
    """
    p = self.params
    text = tf.placeholder(tf.string, shape=[None])
    # [batch, time]
    ids, labels, paddings = self.input_generator.StringsToIds(text)
    chunk_ids = None
    if p.lm.gold_chunks:
      ids, labels, paddings, chunk_ids = lm_inp.LmInput.GetChunks(ids, labels, paddings)
    lengths = tf.reduce_sum(tf.to_int32(1 - paddings), axis=1)
    tokens_from_labels = self.input_generator.IdsToStrings(labels, lengths)
    oovs = tf.equal(labels, self.input_generator.tokenizer.unk_id)
    num_oovs_per_sample = tf.to_int32(
        tf.reduce_sum(tf.to_float(oovs) * (1 - paddings), axis=1))
    # [time, batch]
    ids, paddings, labels, weights, chunk_ids = self._TrimIfPossibleThenTranspose(
        ids, paddings, labels, 1.0 - paddings, chunk_ids)
    batch_size = tf.shape(ids)[1]
    state0 = self.lm.zero_state(batch_size)
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      lower_state0 = self.lm.zero_state(batch_size)
      xent_output, _, _ = self.lm.FPropDefaultTheta(
          inputs=ids,
          paddings=paddings,
          state0=state0,
          lower_state0=lower_state0,
          labels=py_utils.NestedMap(class_ids=labels, class_weights=weights),
          chunk_ids=chunk_ids,
          ids=ids)
    else:
      xent_output, _ = self.lm.FPropDefaultTheta(
          inputs=ids,
          paddings=paddings,
          state0=state0,
          labels=py_utils.NestedMap(class_ids=labels, class_weights=weights),
          chunk_ids=chunk_ids,
          ids=ids)

    per_example_xent = py_utils.HasShape(xent_output.per_example_xent,
                                         tf.shape(ids))
    log_pplx_per_sample = tf.reduce_sum(
        per_example_xent * (1 - paddings), axis=0)
    fetches = {
        'log_pplx_per_token':  # [batch, time]
            tf.transpose(per_example_xent),
        'paddings':  # [batch, time]
            tf.transpose(paddings),
        'lengths':  # [batch]
            lengths,
        'log_pplx_per_sample':  # [batch]
            log_pplx_per_sample,
        'num_oovs_per_sample':  # [batch], int32
            num_oovs_per_sample,
        'tokens_from_labels':  # [batch], string
            tokens_from_labels,
        'ids':  # [batch, time], int32
            ids
    }
    feeds = {'text': text}

    # Also pass intermediate results
    if 'inter_res' in xent_output:
      inter_res = xent_output.inter_res
      for key in inter_res:
        new_key = 'inter_res.%s' %key
        assert new_key not in fetches
        fetches[new_key] = getattr(inter_res, key)
    return fetches, feeds

  def _InferenceSubgraph_RNNStep(self):
    """Inference subgraph for one rnn step.

    Returns:
      fetches: A dictionary of fetches, containing:
        zero_m_out_i: A matrix of shape [batch, output_size].
          m values of the i-th layer of zero recurrent state.
        zero_c_out_i: A matrix of shape [batch, hidden_size].
          c values of the i-th layer of zero recurrent state.
        logits: A matrix of shape [batch, num_candidates]. [i, j]
          is i-th input's j-th candidate's logit.
        m_out_i: A matrix of shape [batch, output_size].
          m values of the i-th layer of new recurrent state after one step.
        c_out_i: A matrix of shape [batch, hidden_size].
          c values of the i-th layer of new recurrent state after one step.
      feeds: A dictionary of feeds, containing:
        step_ids: A matrix of shape [batch, 1]. [i, 0]
          is the word id to run one step for the i-th input.
        candidate_ids: A 3D tensor of shape [batch, num_candidates, 2].
          [i, j, 0] = i just for indexing convenience.
          [i, j, 1] is the word id of the i-th input's j-th candidate.
        m_in_i: A matrix of shape [batch, output_size].
          m values of input recurrent state.
        c_in_i: A matrix of shape [batch, hidden_size].
          c values of input recurrent state.
    """
    fetches, feeds = {}, {}

    # Run one step with input ids and return logits.
    # [batch, 1]
    step_ids = tf.placeholder(tf.int32, [None, 1])
    feeds['step_ids'] = step_ids

    # Return logits only for certain candidate ids. This is to avoid returning
    # a big list of logits for all words.
    # This is a 3D tensor and it satisfies that:
    #    candidate_ids[i, j, 0] = i (just for indexing convenience)
    #    candidate_ids[i, j, 1] = the word id of the j-th candidate
    # [batch, num_candidates, 2]
    candidate_ids = tf.placeholder(tf.int32, [None, None, 2])
    feeds['candidate_ids'] = candidate_ids

    # Get initial zero states.
    batch_size = tf.shape(step_ids)[0]
    zero_state = self.lm.zero_state(batch_size)

    # Input LM state.
    state0 = zero_state.Transform(lambda x: tf.placeholder(tf.float32))

    # Run LM for one step
    step_ids_vec = tf.reshape(step_ids, [-1])
    step_paddings = tf.zeros(tf.shape(step_ids_vec), dtype=self.params.dtype)

    p = self.params
    lower_state0 = None
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      lower_zero_state0 = self.lm.lower_rnns.zero_state(batch_size)
      lower_state0 = lower_zero_state0.Transform(lambda x: tf.placeholder(tf.float32))
    res = self.lm.Step(self.lm.theta, step_ids_vec, step_paddings,
                               state0, lower_state0=lower_state0, step_inference=True) # TODO(jmluo) HACKY
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      out, state1, lower_state1 = res
      # add more feeds and fetches for lower level rnn
      feeds['lowerrnnstate:m'] = lower_state0.rnn[0].m
      feeds['lowerrnnstate:c'] = lower_state0.rnn[0].c
      fetches['lowerrnnstate:m'] = lower_state1.rnn[0].m
      fetches['lowerrnnstate:c'] = lower_state1.rnn[0].c
    else:
      out, state1 = res

    # Create feeds/fetches map for states.
    for i, (zero_s, s0, s1) in enumerate(
        zip(zero_state.rnn, state0.rnn, state1.rnn)):
      feeds['rnnstate:m_%02d' % i] = s0.m
      feeds['rnnstate:c_%02d' % i] = s0.c
      fetches['rnnstate:zero_m_%02d' % i] = zero_s.m
      fetches['rnnstate:zero_c_%02d' % i] = zero_s.c
      fetches['rnnstate:m_%02d' % i] = s1.m
      fetches['rnnstate:c_%02d' % i] = s1.c


    # Collect logits for candidates
    # [batch, num_candidates]
    prob = tf.nn.softmax(out.logits)
    candidate_prob = tf.gather_nd(prob, candidate_ids)
    candidate_logits = tf.log(candidate_prob)
    fetches['logits'] = candidate_logits


    if 'gating_probs' in out:
      fetches['gating_probs'] = out.gating_probs
    if 'cce' in out:
      fetches['cce'] = out.cce

    # print('check here', fetches)
    return fetches, feeds
