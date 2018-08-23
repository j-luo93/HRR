from __future__ import division

import os
import codecs
import numpy as np

from collections import Counter

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

class Vocab(object):

    # TODO add frequency filtering
    def __init__(self, path, size):
        if os.path.isfile(path):
            vocab = list()
            with codecs.open(path, 'r', 'utf8') as fin:
                for line in fin:
                    w, c = line.strip().split('\t')
                    vocab.append(w)
        else:
            raise ValueError('cannot find vocab file at %s' %path)
        if size > 0:
            vocab = vocab[:size]
        self.w2i = dict(zip(vocab, range(len(vocab))))
        self.i2w = {i: w for w, i in self.w2i.items()}

    def indexify(self, sent):
        return map(lambda token: self.w2i.get(token, UNK_ID), sent)

    def __len__(self):
        return len(self.w2i)

class Datasets(object):

    def __init__(self, **kwargs):
        self.data_dir = kwargs['data_dir']
        self.vocab_size = kwargs['vocab_size']
        train_path = self.data_dir + '/train.txt'
        dev_path = self.data_dir + '/dev.txt'
        test_path = self.data_dir + '/test.txt'
        vocab_path = self.data_dir + '/vocab.txt'
        buckets = [10, 20, 30, 40, 50]

        train_dataset = Dataset(train_path)
        dev_dataset = Dataset(dev_path)
        test_dataset = Dataset(test_path)

        self.vocab = Vocab(vocab_path, self.vocab_size)
        self.data = {'train': train_dataset, 'dev': dev_dataset, 'test': test_dataset}
        for split, dataset in self.data.items():
            dataset._indexify(self.vocab)
            if split == 'train':
                dataset._bucketify(buckets)

    def __len__(self):
        return len(self.data['train'])

    def get_random_batch(self, batch_size):
        return self.data['train'].get_random_batch(batch_size)

class Batch(object):

    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets

class Dataset(object):

    def __init__(self, path):
        self.indexified = False
        self.bucketified = False
        self.data = list()
        with codecs.open(path, 'r', 'utf8') as fin:
            for line in fin:
                sent = line.strip().split(' ')
                self.data.append(sent)

    def __len__(self):
        if self.bucketified:
            return sum(map(len, self.data))
        else:
            return len(self.data)

    def get_random_batch(self, batch_size):
        assert self.indexified
        assert self.bucketified
        bucket_id = np.random.choice(len(self.bucket_probs), p=self.bucket_probs)
        batch_ids = np.random.choice(len(self.data[bucket_id]), batch_size)
        batch = self._get_batch(bucket_id, batch_ids)
        return batch

    def _get_batch(self, bucket_id, batch_ids):
        raw = [self.data[bucket_id][i] for i in batch_ids]
        max_l = max(map(len, raw)) + 1
        ids = list()
        targets = list()

        def pad(seq, max_l):
            l = len(seq)
            assert l <= max_l
            return seq + [PAD_ID] * (max_l - l)

        for d in raw:
            ids.append(pad([SOS_ID] + d, max_l))
            targets.append(pad(d + [EOS_ID], max_l))
        return Batch(np.asarray(ids), np.asarray(targets))

    def __iter__(self):
        for sent in self.data:
            yield sent

    def _indexify(self, vocab):
        assert not self.indexified
        indexified_data = list()
        for sent in self.data:
            indexified_data.append(vocab.indexify(sent))
        self.data = indexified_data
        self.indexified = True

    def _bucketify(self, buckets):
        assert not self.bucketified
        bucketified_data = [list() for _ in range(len(buckets))]

        def get_bucket_id(length):
            for i, l in enumerate(buckets):
                if l >= length:
                    return i
            return -1

        for sent in self.data:
            bucket_id = get_bucket_id(len(sent))
            if bucket_id != -1:
                bucketified_data[bucket_id].append(sent)
        self.data = bucketified_data
        num_sents = np.asarray(map(len, self.data)).astype('float32')
        self.bucket_probs = num_sents / num_sents.sum()
        self.bucketified = True
