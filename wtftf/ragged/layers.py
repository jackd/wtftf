import tensorflow as tf

from wtftf.meta import layered

# factories
from_row_splits = layered(tf.RaggedTensor.from_row_splits)
from_nested_row_splits = layered(tf.RaggedTensor.from_nested_row_splits)
from_value_rowids = layered(tf.RaggedTensor.from_value_rowids)
from_nested_value_rowids = layered(tf.RaggedTensor.from_nested_value_rowids)
from_row_lengths = layered(tf.RaggedTensor.from_row_lengths)
from_nested_row_lengths = layered(tf.RaggedTensor.from_nested_row_lengths)
from_uniform_row_length = layered(tf.RaggedTensor.from_uniform_row_length)
from_row_limits = layered(tf.RaggedTensor.from_row_limits)
from_row_starts = layered(tf.RaggedTensor.from_row_starts)
from_sparse = layered(tf.RaggedTensor.from_sparse)
from_tensor = layered(tf.RaggedTensor.from_tensor)

if tf.version.VERSION < "2.4":
    # attrs
    values = layered(tf.RaggedTensor.values.fget)
    row_splits = layered(tf.RaggedTensor.row_splits.fget)
    flat_values = layered(tf.RaggedTensor.flat_values.fget)
    nested_row_splits = layered(tf.RaggedTensor.nested_row_splits.fget)

    # methods
    row_lengths = layered(tf.RaggedTensor.row_lengths)
    row_starts = layered(tf.RaggedTensor.row_starts)
    value_rowids = layered(tf.RaggedTensor.value_rowids)
    to_tensor = layered(tf.RaggedTensor.to_tensor)
    nrows = layered(tf.RaggedTensor.nrows)

else:
    # attrs

    def values(self):
        return self.values

    def row_splits(self):
        return self.row_splits

    def flat_values(self):
        return self.flat_values

    def nested_row_splits(self):
        return self.nested_row_splits

    # methods

    def row_lengths(self):
        return self.row_lengths()

    def row_starts(self):
        return self.row_starts()

    def value_rowids(self):
        return self.value_rowids()

    def to_tensor(self, *args, **kwargs):
        return self.to_tensor(*args, **kwargs)

    def nrows(self):
        return self.nrows()
