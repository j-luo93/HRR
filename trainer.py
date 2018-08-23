from __future__ import division, print_function

import tensorflow as tf

class Trainer(object):

    def __init__(self, **kwargs):
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']

    def train(self, model, datasets):
        with tf.Session() as sess:

            # TODO add saving and loading support
            max_step = len(datasets) // self.batch_size * self.num_epochs
            sess.run(tf.global_variables_initializer())
            for step in range(max_step):
                batch = datasets.get_random_batch(self.batch_size)
                feed_dict = {'input_ids:0': batch.ids, 'target_ids:0': batch.targets}
                _, step_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                print('step_loss: %.4f' %step_loss)
