from __future__ import division, print_function

import sys
import tensorflow as tf

from utils import Map
from datasets import DatasetIterator

class Tracker(object):

    def __init__(self, path, keys):
        self._initializer = {k: 0.0 for k in keys}
        self._writer = tf.summary.FileWriter(path)
        self.clear()
    
    def clear(self):
        self.values = Map(**self._initializer)
        self.weights = Map(**self._initializer)

    def update(self, key, value, weight):
        assert key in self.values
        self.values[key] = self.values[key] + weight * value
        self.weights[key] = self.weights[key] + weight

    def add_summary(self, summary, global_step=None):
        self._writer.add_summary(summary, global_step=global_step)

    def add_graph(self, graph):
        self._writer.add_graph(graph)

    def print(self, msg=''):
        print('\n' + msg)

        for key in self.values:
            v = self.values[key]
            w = self.weights[key]
            print('%s\t%.4f' %(key, v / w))
        self.clear()

class Trainer(object):

    def __init__(self, **kwargs):
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.print_interval = kwargs['print_interval']
        self.eval_interval = kwargs['eval_interval']
        self.log_dir = kwargs['log_dir']
        self.train_tracker = Tracker(self.log_dir + '/train', ['step_loss'])
        self.dev_tracker = Tracker(self.log_dir + '/dev', ['step_loss'])
        self.test_tracker = Tracker(self.log_dir + '/test', ['step_loss'])

    def train(self, model, datasets):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.train_tracker.add_graph(sess.graph)

            # TODO add saving and loading support
            max_step = len(datasets) // self.batch_size * self.num_epochs
            sess.run(tf.global_variables_initializer())
            for step in range(1, max_step + 1):
                batch = datasets.get_random_batch(self.batch_size)
                feed_dict = {'input_ids:0': batch.ids, 'target_ids:0': batch.targets}
                _, step_loss, num_words, summary = sess.run([model.train_op, model.loss, model.num_words, model.summary_op], feed_dict=feed_dict)

                self.train_tracker.update('step_loss', step_loss, num_words)
                self.train_tracker.add_summary(summary, step)

                print('\rstep %d' %step, end='')
                sys.stdout.flush()
                
                if step % self.print_interval == 0:
                    self.train_tracker.print(msg='train metrics:')    

                if step % self.eval_interval == 0:
                    for split in ['dev', 'test']:
                        dataset = datasets[split]
                        iterator = DatasetIterator(dataset, self.batch_size)
                        tracker = self.dev_tracker if split == 'dev' else self.test_tracker
                        for batch in iterator:
                            feed_dict = {'input_ids:0': batch.ids, 'target_ids:0': batch.targets}
                            step_loss, num_words, summary = sess.run([model.loss, model.num_words, model.summary_op], feed_dict=feed_dict)
                            tracker.update('step_loss', step_loss, num_words)
                            tracker.add_summary(summary)
                        tracker.print(msg='%s metrics:' %split)


                    
