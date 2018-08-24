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


def create_scalar_summary(key, value):
	return tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])

def initialize_op(variable):
    value = tf.initializers.random_uniform(-0.08, 0.08)(variable.get_shape())
    op = tf.assign(variable, value)
    return op

def ndim(tensor):
    return len(tensor.get_shape())


def circular_conv(a, b):
    with tf.name_scope('circular_conv'):
        a_fft = tf.fft(tf.complex(a, 0.0))
        b_fft = tf.fft(tf.complex(b, 0.0))
        ifft = tf.ifft(a_fft * b_fft)
        res = tf.cast(tf.real(ifft), 'float32')
    return res

def circular_corr(a, b):
    with tf.name_scope('circular_corr'):
        a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
        b_fft = tf.fft(tf.complex(b, 0.0))
        ifft = tf.ifft(a_fft * b_fft)
        res = tf.cast(tf.real(ifft), 'float32')
    return res
