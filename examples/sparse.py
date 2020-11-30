import tensorflow as tf

from wtftf.sparse import layers as sparse_layers

st = tf.keras.Input((None,), sparse=True)
rhs = tf.keras.Input((None,))

i, j = tf.unstack(sparse_layers.indices(st), axis=-1)
values = sparse_layers.values(st)

gathered = tf.gather(rhs, j)
out = tf.math.segment_sum(gathered * values, i)

model = tf.keras.Model((st, rhs), out)
