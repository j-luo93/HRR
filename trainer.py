from __future__ import division, print_function

import sys
import tensorflow as tf

from utils import Map, create_scalar_summary, ndim, initialize_op
from datasets import DatasetIterator

class Tracker(object):

    def __init__(self, path):
        self._writer = tf.summary.FileWriter(path)
        self.clear()
    
    def clear(self):
        self.values = Map()
        self.weights = Map()

    def update(self, key, value, weight):
        if key not in self.values:
            self.values[key] = self.weights[key] = 0.0
        self.values[key] = self.values[key] + weight * value
        self.weights[key] = self.weights[key] + weight

    def add_summary(self, summary, global_step=None):
        self._writer.add_summary(summary, global_step=global_step)

    def add_graph(self, graph):
        self._writer.add_graph(graph)

    def check(self, msg=''):
        print('\n' + msg)

        for key in self.values:
            v = self.values[key]
            w = self.weights[key]
            print('%s\t%.4f' %(key, v / w))
        self.clear()

    def get_stat(self, key):
        v = self.values[key]
        w = self.weights[key]
        return v / w

class Trainer(object):

    def __init__(self, **kwargs):
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.check_interval = kwargs['check_interval']
        self.eval_interval = kwargs['eval_interval']
        self.log_dir = kwargs['log_dir']
        self.train_tracker = Tracker(self.log_dir + '/train')
        self.dev_tracker = Tracker(self.log_dir + '/dev')
        self.test_tracker = Tracker(self.log_dir + '/test')

    def train(self, model, datasets):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.train_tracker.add_graph(sess.graph)

            # TODO add saving and loading support
            max_step = len(datasets) // self.batch_size * self.num_epochs
            sess.run(tf.global_variables_initializer())
            
            # customized initializer
            init_ops = list()
            for v in tf.trainable_variables():
                if ndim(v) == 2:
                    init_ops.append(initialize_op(v))
            sess.run(init_ops)

            # fetches are the same for all steps
            op_names = ['_', 'summary']
            ops_to_run = [model.train_op, model.summary_op]
            for name, (metric, weight) in model.metrics.items():
                ops_to_run.extend([metric, weight])
                op_names.extend([name, name + '_weight'])
                
            for step in range(1, max_step + 1):
                batch = datasets.get_random_batch(self.batch_size)
                
                # prepare feeds
                feed_dict = {input_: getattr(batch.inputs, name) for name, input_ in model.inputs.items()}
                
                # run session and wrap the results into a dict
                res = sess.run(ops_to_run, feed_dict=feed_dict)
                res = dict(zip(op_names, res))
                
                # update metrics
                for name in model.metrics:
                    self.train_tracker.update(name, res[name], res[name + '_weight'])
                # add sumamry
                self.train_tracker.add_summary(res['summary'], step)
                

                print('\rstep %d' %step, end='')
                sys.stdout.flush()
                
                if step % self.check_interval == 0:
                    self.train_tracker.check(msg='train metrics:')    

                if step % self.eval_interval == 0:
                    for split in ['dev', 'test']:
                        dataset = datasets[split]
                        iterator = DatasetIterator(dataset, self.batch_size)
                        tracker = self.dev_tracker if split == 'dev' else self.test_tracker
                        for batch in iterator:
                            feed_dict = {input_: getattr(batch.inputs, name) for name, input_ in model.inputs.items()}
                            # do NOT run model.train_op
                            res = sess.run(ops_to_run[1:], feed_dict=feed_dict) 
                            res = dict(zip(op_names[1:], res))
                            
                            for name in model.metrics:
                                tracker.update(name, res[name], res[name + '_weight'])
                        for name in model.metrics:
                            tracker.add_summary(create_scalar_summary(name, tracker.get_stat(name)), step)
                        tracker.check(msg='%s metrics:' %split)


                    
