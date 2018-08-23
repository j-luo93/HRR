import tensorflow as tf

def mm3by2(a, b, **kwargs):
    tf.assert_rank(a, 3)
    tf.assert_rank(b, 2)

    d1, d2, d3 = a.get_shape()
    return tf.matmul(tf.reshape(a, [-1, d3]), b, **kwargs)

class Map(dict):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self.__dict__ = self
