from __future__ import division

import os
import codecs
import numpy as np

from collections import Counter

from utils import Map

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

    def __getitem__(self, key):
        return self.data[key]

class DatasetIterator(object):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        assert not dataset.bucketified and dataset.indexified
        self.batch_size = batch_size
        self.next_batch_id = 0

    def __iter__(self):
        max_id = len(self.dataset)
        while self.next_batch_id < max_id:
            batch_ids = range(self.next_batch_id, min(self.next_batch_id + self.batch_size, max_id))
            self.next_batch_id = batch_ids[-1] + 1
            yield self.dataset._get_batch(None, batch_ids)
        
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
        if bucket_id is None:
            raw = [self.data[i] for i in batch_ids]
        else:
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
        batch = Map(ids=ids, targets=targets)
        return batch

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
