import tensorflow as tf

def mm3by2(a, b, **kwargs):
    tf.assert_rank(a, 3)
    tf.assert_rank(b, 2)

    d1, d2, d3 = a.get_shape()
    return tf.matmul(tf.reshape(a, [-1, d3]), b, **kwargs)
