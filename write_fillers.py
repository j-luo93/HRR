from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import numpy as np

import tensorflow as tf


from collections import Counter
import codecs

from tensorflow.python import pywrap_tensorflow


def print_tensors_in_checkpoint_file(file_name,
                                     tensor_name,
                                     all_tensors,
                                     all_tensor_names=False):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)
        if all_tensors:
          print(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      return reader.get_tensor(tensor_name)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))

def get_predictor(path, suffix):
  import numpy as np
  import google3
  from google3.learning.brain.research.babelfish.predictor import Predictor
  
  p = Predictor('%s/inference_graphs/inference.pbtxt' %path, subgraph_name='default', device_type='cpu')
  p.Load(checkpoint='%s/train/%s' %(path, suffix))
  
  p_step = Predictor('%s/inference_graphs/inference.pbtxt' %path, subgraph_name='rnn_step', device_type='cpu')
  p_step.Load(checkpoint='%s/train/%s' %(path, suffix))
  
  return p, p_step

def get_vocab(vocab_file):

    vocab = list()
    with codecs.open(vocab_file, 'r', 'utf8') as fin:
      for line in fin:
        vocab.append(line.strip().split('\t')[0])
    return vocab
  
def run(predictor, state0, inp_ind, vocab_range, num_layers=None):
  if state0 is None:
    assert num_layers is not None
    import numpy as np
    state0 = [(np.zeros([1, 512]), np.zeros([1, 512])) for _ in xrange(num_layers)]
    
  inp = [[inp_ind]]
  feed_dict = {'step_ids':inp, 'candidate_ids': [[[0, i] for i in range(vocab_range)]]}
  fetch_list = ['logits']
  for i, (m, c) in enumerate(state0):
    m_name = 'rnnstate:m_%02d' %i
    c_name = 'rnnstate:c_%02d' %i
    feed_dict[m_name] = m
    feed_dict[c_name] = c
    fetch_list.append(m_name)
    fetch_list.append(c_name)
  try:
    fetch_list.append('gating_probs')
    res = predictor.Run(fetch_list, **feed_dict)
    gating = True
  except:
    res = predictor.Run(fetch_list[:-1], **feed_dict)
    gating = False
  
  state1 = list()
  for i in xrange((len(res) - 1) // 2):
    state1.append((res[2 * i + 1], res[2 * i + 2]))
  if gating:
    return res[0], state1, res[-1]
  else:
    return res[0], state1
  
  
def prepare_all(file_name, vocab_file, vocab_range, num_fillers, model='1b', separate_biases=False, mode='basic'):
  assert mode in ['basic', 'rs', 'dec_only', 'baseline']
  import numpy as np
  
  if separate_biases:
    b = [print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/softmax/bias_r%d_0/var' %(model, role_ind), all_tensors=False) for role_ind in xrange(rf.shape[1])]
  else:
    b = None
  vocab = get_vocab(vocab_file)[:vocab_range]
  w2i = {w: i for w, i in zip(vocab, range(len(vocab)))}

  if mode == 'rs':
    rs = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/rs/var_0/var' %model, all_tensors=False)[:vocab_range]
    rR = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/rR/var' %model, all_tensors=False)
    s = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/s/var_0/var' %model, all_tensors=False)
    F = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/F/var' %model, all_tensors=False)

    rs = np.reshape(rs, [-1, 2, 2])
    r = np.matmul(rs, rR)[:vocab_range]
    f = (np.reshape(s[:vocab_range], [vocab_range, 2, num_fillers, 1]) * F).sum(axis=2) # 10000 x 2 x 512
    rf = circular_conv(r, f) # 10000 x 2 x 512
    e = rf.sum(axis=1) # 10000 x 512

    true_f = circular_corr(rR, np.reshape(e, [vocab_range, 1, -1]))
    return Params(s=s, r=r, F=F, hid_f=f, rf=rf, e=e, b=b, rs=rs, rR=rR, f=true_f, w2i=w2i)
  elif mode == 'basic':
    total = 0
    all_s = list()
    while True:
      s = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/s/var_%d/var' %(model, len(all_s)), all_tensors=False)
      all_s.append(s)
      total += len(s)
      if total >= vocab_range:
          break
    s = np.concatenate(all_s, axis=0)
    F = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/F/var' %model, all_tensors=False)
    r = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/r/var' %model, all_tensors=False)
    
    assert len(F.shape) == 3
    nr = F.shape[0]
    _f = list()
    for i in range(nr):
        fi = np.matmul(s[:vocab_range].reshape([vocab_range, nr, num_fillers])[:, i], F[i])
        _f.append(fi)
    f = np.stack(_f, axis=1)
    if args.normalize:
        f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
    #f = (np.reshape(s[:vocab_range], [vocab_range, 2, num_fillers, 1]) * F).sum(axis=2) # 10000 x 2 x 512
    rf = circular_conv(r, f) # 10000 x 2 x 512
    e = rf.sum(axis=1) # 10000 x 512
    
    return Params(s=s, r=r, F=F, f=f, rf=rf, e=e, b=b, w2i=w2i)
  elif mode == 'baseline':
    total = 0
    es = list()
    while True:
      e = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/var_%d/var' %(model, len(es)), all_tensors=False)
      es.append(e)
      total += len(e)
      if total >= vocab_range:
          break
    e = np.concatenate(es, axis=0)
    return Params(e=e, w2i=w2i) 

  else:
    e = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/e_l/var_0/var' %model, all_tensors=False)[:vocab_range]
    r = print_tensors_in_checkpoint_file(file_name, tensor_name='%s_word_level_lm/lm/emb/r/var' %model, all_tensors=False)
    f = circular_corr(r, np.reshape(e, [vocab_range, 1, -1]))
    return Params(r=r, e=e, f=f, b=b, w2i=w2i)

def circular_conv(a, b):
  a = np.fft.fft(a)
  b = np.fft.fft(b)
  return np.real(np.fft.ifft(a * b))

def circular_corr(a, b):
  a = np.conj(np.fft.fft(a))
  b = np.fft.fft(b)
  return np.real(np.fft.ifft(a * b))

def NN(vec, vec_src=None, w2i=None, size=10):
  assert vec_src is not None, 'pass it!'
  assert w2i is not None, 'pass it!'
  from scipy.spatial.distance import cosine
  dist = dict()
  for w in w2i:
    vec_other = vec_src[w2i[w]]
    dist[w] = '%.4f.' %(1.0 - cosine(vec, vec_other))
  
  from operator import itemgetter
  return sorted(dist.items(), key=itemgetter(1), reverse=True)[:size]

def print_NNs(vecs, vec_srcs, w2i, size=10):
  for i in xrange(len(vecs)):
    v = vecs[i]
    vs = vec_srcs[:, i]
    print(NN(v, vec_src=vs, w2i=w2i, size=size))

def decode(h, r):
  res = list()
  for ri in r:
    res.append(circular_corr(ri, h)[0])
  return res

class Params(dict):
  
  def __init__(self, *args, **kwargs):
    super(Params, self).__init__(*args, **kwargs)
    self.__dict__ = self
    
def corr(f_noisy, f, mode='pearson', b=None):
  assert mode in ['pearson', 'spearman']
  from scipy.stats import pearsonr, spearmanr
  
  dot0 = np.dot(f[:, 0, :], f_noisy[0])
  dot1 = np.dot(f[:, 1, :], f_noisy[1])
  
  if b is not None:
    assert len(b) == 2
    size = len(dot0)
    dot0 += b[0][:size]
    dot1 += b[1][:size]
    
  if mode == 'pearson':
    print(pearsonr(dot0, dot1))
    print(pearsonr(np.exp(dot0), np.exp(dot1)))
  else:
    print(spearmanr(dot0, dot1))
    print(spearmanr(np.exp(dot0), np.exp(dot1)))
  
  p0 = softmax(dot0)
  p1 = softmax(dot1)
  print(KL(p0, p1))
  print(KL(p1, p0))
    
def softmax(a):
  M = np.max(a, axis=-1)
  e = np.exp(a - M)
  return e / e.sum(axis=-1, keepdims=True)
  
class Model(object):
  
  def __init__(self, path, suffix, vocab_file, vocab_size, num_fillers, model_name='1b', separate_biases=False, mode='basic'):
    ckpt_file = '%s/%s' %(path, suffix)
    self.params = prepare_all(ckpt_file, vocab_file, vocab_size, num_fillers, model=model_name, separate_biases=separate_biases, mode=mode)
#     p, p_step = get_predictor(path, suffix)
#     self.p = p
#     self.p_step = p_step
  
  def run_step_seq(self, text, num_layers=2):
    return run_seq(self.p_step, self.params, text, num_layers=num_layers)
   
 
    
def run_seq(predictor, params, text, num_layers=2):
  inputs = ['<S>'] + text.split()
  res = list()
  size = len(params.w2i)
  for inp in inputs:
    if inp == '<S>':
      ret = run(predictor, None, params.w2i[inp], size, num_layers=num_layers)
    else:
      ret = run(predictor, state, params.w2i[inp], size)
    if len(ret) == 2:
      _, state = ret
      gp = None
    else:
      _, state, gp = ret
    m = state[-1][0]
    f_noisy = decode(m, params.r)
    res.append((f_noisy, gp))
  return res
  
def corr_seq(f_noisy_lst, f, mode='pearson', b=None):
  for f_noisy in f_noisy_lst:
    corr(f_noisy, f, mode=mode, b=b)
    
def KL(p, q):
  from scipy.stats import entropy
  return entropy(p, q)

def get_models(prefix, timestamps, ckpt_nums, vocab_file, vocab_size, num_fillers_per_role, model_name, separate_biases=False):
  assert len(timestamps) == len(ckpt_nums)
  models = list()
  for ts, num in zip(timestamps, ckpt_nums):
    path = prefix + ts
    suffix = 'ckpt-%08d' %num
    model = Model(path, suffix, vocab_file, vocab_size, num_fillers_per_role, model_name, separate_biases=separate_biases)
    models.append(model)
  return models  

def run_pipeline(model, text, num_layers):
  f_noisy = model.run_step_seq(text, num_layers=num_layers)
  f_noisy, gps = zip(*f_noisy)
  corr_seq(f_noisy, model.params.f)
  for noisy, gp in zip(f_noisy, gps):
    print_NNs(noisy, model.params.f, model.params.w2i)
    print(gp)


def write(model, prefix, size=2500, model_name='ptb', vocab_size=9978, key='f', decomp=True):
  def to_string(x):
    return map(lambda xx: '%.4f' %xx, x)
  
  f = getattr(model.params, key)
  indices = [model.params.w2i[w] for w in filter(lambda w: w != '<unk>', lst[:vocab_size])]
  #indices = [model.params.w2i.get(w, 3) for w in lst[:vocab_size]]
  
  trans = {v: k for k, v in model.params.w2i.items()}
#   indices = range(size)
#   trans = {v: k for k, v in model.params.w2i.items()}
  if decomp:
    f = np.reshape(f, [vocab_size, 2, -1])
    f1 = f[indices, 0, :]
    f2 = f[indices, 1, :]
  else:
    f = f[indices]

  if decomp:
    with codecs.open('saved_embeddings/%s/%s.roles' %(model_name, prefix), 'w', 'utf8') as fr:
      r = model.params.r
      nr, d = r.shape
      fr.write('%d %d\n' %(nr, d))
      for ri in range(nr):
        fr.write('%s %s\n' %('role-%d' %ri, ' '.join(to_string(r[ri]))))


  with codecs.open('saved_embeddings/%s/%s-v-%d' %(model_name, prefix, size), 'w', 'utf8') as fv:
    fv.write('word\tset\tnorm\n')
  
    with codecs.open('saved_embeddings/%s/' %model_name + prefix + '.tsv', 'w', 'utf8') as fout:
      with codecs.open('saved_embeddings/%s/' %model_name + prefix + '.w2v', 'w', 'utf8') as fw2v:
        if decomp:
          for i in xrange(size):
            fout.write('%s\n' %('\t'.join(to_string(f1[i]))))
            fv.write('%s\tf1\t%.4f\n' %(trans[indices[i]], np.linalg.norm(f1[i])))
        

          for i in xrange(size):
            fout.write('%s\n' %('\t'.join(to_string(f2[i]))))
            fv.write('%s\tf2\t%.4f\n' %(trans[indices[i]], np.linalg.norm(f2[i])))
        else:
          for i in range(size):
            fout.write('%s\n' %('\t'.join(to_string(f[i]))))
            fv.write('%s\te\t%.4f\n' %(trans[indices[i]], np.linalg.norm(f[i])))

        actual_vocab_size = size #len(indices)
        dim = f1.shape[-1] if decomp else f.shape[-1]
        multiplier = 2 if decomp else 1
        fw2v.write('%d %d\n' %(actual_vocab_size * multiplier, dim))
        for i in xrange(actual_vocab_size):
          try:
            if decomp:
              fw2v.write('%s %s\n' %(trans[indices[i]] + '-f1', ' '.join(to_string(f1[i])) ))
              fw2v.write('%s %s\n' %(trans[indices[i]] + '-f2', ' '.join(to_string(f2[i])) ))
            else:
              fw2v.write('%s %s\n' %(trans[indices[i]], ' '.join(to_string(f[i])) ))

          except:
            print(i)
            1/0
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_train_data_path', '-ptdp', help='Used to extract most frequent words')
    parser.add_argument('--word_list', '-wl', help='Word list')
    parser.add_argument('--model_path', '-mp', help='Model path')
    parser.add_argument('--checkpoint', '-ckpt', type=int, help='Checkpoint number')
    parser.add_argument('--vocab_path', '-vp', help='Vocab path')
    parser.add_argument('--dim', '-d', type=int, help='Dimensionality of embeddings')
    parser.add_argument('--write_size', '-ws', type=int, help='How many words to write to file')
    parser.add_argument('--vocab_size', '-vs', type=int, help='How many words to read as vocab')
    parser.add_argument('--output_prefix', '-op', help='Output file prefix')
    parser.add_argument('--output_directory', '-od', help='Where to save the output files')
    parser.add_argument('--mode', '-m', help='Can be baseline, e, or f')
    parser.add_argument('--normalize', '-n', action='store_true', help='Normalize f embeddings to get e embeddings')
    args = parser.parse_args()

    assert args.mode in ['baseline', 'e', 'f']
    #ptb_path = "HRR/data/ptb-chunk/train.txt"
    #ptb_path = "HRR/data/ptb/train.txt"
    if args.word_list is not None:
        lst = [l.strip() for l in codecs.open(args.word_list, 'r', 'utf8')]
    else:
        cnt = Counter()                                                              
        with codecs.open(args.ptb_train_data_path, 'r', 'utf8') as fin:
          for line in fin:
            for w in line.strip().split():
              cnt[w] += 1
        lst = [x for x, y in sorted(cnt.items(), key=lambda x: x[1], reverse=True)]

    # model path
    #path = 'train/ptb-full-NF250-fixed-decay/train'
    #path = 'train/ptb-baseline-fixed-decay/train'
    # checkpoint number
    suffix = 'ckpt-%08d' %args.checkpoint
    #suffix = 'ckpt-00003285'
    #vocab_file = 'HRR/data/ptb-chunk/vocab.txt'
    #vocab_file = 'HRR/data/ptb/vocab.txt'

    if args.mode != 'baseline':
        mode = 'basic'
    else:
        mode = 'baseline'

    assert args.vocab_size >= args.write_size
    model_ptb = Model(args.model_path, suffix, args.vocab_path, args.vocab_size, args.dim, model_name=args.output_directory, mode=mode)
    #model_ptb = Model(path, suffix, vocab_file, vocab_size, 50, model_name='ptb', mode='baseline')

    if args.mode == 'f':
        key = 'f'
        decomp = True
    else:
        key = 'e'
        decomp = False

    write(model_ptb, args.output_prefix, size=args.write_size, vocab_size=args.vocab_size, model_name=args.output_directory, key=key, decomp=decomp)
    #write(model_ptb, 'ptb-full-NF250-e-fixed-decay', vocab_size=vocab_size, model_name='ptb-chunk', key='e', decomp=False)
    #write(model_ptb, 'ptb-full-NF250-fixed-decay', vocab_size=vocab_size, model_name='ptb-chunk')
    #write(model_ptb, 'ptb-baseline-fixed-decay', vocab_size=vocab_size, model_name='ptb', key='e', decomp=False)
