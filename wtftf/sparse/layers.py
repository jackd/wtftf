import tensorflow as tf

from wtftf.meta import layered


def _sparse_tensor(args, **kwargs):
    return tf.SparseTensor(*args, **kwargs)


# constructors
SparseTensor = layered(tf.SparseTensor)

# attrs
values = layered(tf.SparseTensor.values.fget)
indices = layered(tf.SparseTensor.indices.fget)
dense_shape = layered(tf.SparseTensor.dense_shape.fget)
