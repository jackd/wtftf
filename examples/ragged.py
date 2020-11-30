import tensorflow as tf

from wtftf.ragged import layers as rl

rt = tf.keras.Input((None,), ragged=True)
values = rl.values(rt)
row_ids = rl.value_rowids(rt)
seg_sum = tf.math.segment_sum(values, row_ids)

model = tf.keras.Model(rt, seg_sum)
print(model(tf.ragged.range(([2, 3]))))
