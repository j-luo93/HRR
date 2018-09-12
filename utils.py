import tensorflow as tf

def mm3by2(a, b, **kwargs):
    tf.assert_rank(a, 3)
    tf.assert_rank(b, 2)

    d1, d2, d3 = tf.unstack(tf.shape(a))
    return tf.matmul(tf.reshape(a, [-1, d3]), b, **kwargs)

def gather_last_d(params, indices):
    bs, sl, v = tf.unstack(tf.shape(params))
    bs_ind = tf.tile(tf.reshape(tf.range(bs), [bs, 1]), [1, sl])
    sl_ind = tf.tile(tf.reshape(tf.range(sl), [1, sl]), [bs, 1])
    indices = tf.reshape(indices, [bs, sl])
    indices = tf.stack([bs_ind, sl_ind, indices], axis=-1)
    return tf.gather_nd(params, indices)

class Map(dict):

    def __init__(self, default=None, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_scalar_summary(key, value):
	return tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])

def initialize_op(variable):
    value_init = tf.initializers.random_uniform(-0.08, 0.08)
    value = value_init(variable.get_shape())
    op = tf.assign(variable, value)
    return op

def get_name(name, default_name):
    assert name or default_name
    return name if name else default_name

def ndim(tensor):
    return len(tensor.get_shape())

def circular_conv(a, b, name=''):
    name = get_name(name, 'circular_conv')
    with tf.name_scope(name):
        a_fft = tf.fft(tf.complex(a, 0.0))
        b_fft = tf.fft(tf.complex(b, 0.0))
        ifft = tf.ifft(a_fft * b_fft)
        res = tf.cast(tf.real(ifft), 'float32')
    return res

def circular_corr(a, b, name=''):
    name = get_name(name, 'circular_corr')
    with tf.name_scope(name):
        a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
        b_fft = tf.fft(tf.complex(b, 0.0))
        ifft = tf.ifft(a_fft * b_fft)
        res = tf.cast(tf.real(ifft), 'float32')
    return res
