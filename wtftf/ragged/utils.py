import functools

import tensorflow as tf

from wtftf.meta import memoized_property
from wtftf.ragged import layers

IntTensor = tf.Tensor


@functools.wraps(tf.RaggedTensor.ragged_rank)
def ragged_rank(rt) -> int:
    """ragged_rank from `RaggedTensor` or `KerasTensor` equivalent."""
    if isinstance(rt, tf.RaggedTensor):
        return rt.ragged_rank

    assert tf.keras.backend.is_keras_tensor(rt)
    type_spec = rt.type_spec
    assert isinstance(type_spec, tf.RaggedTensorSpec)
    return type_spec.ragged_rank


def is_ragged(x) -> bool:
    """Indicate if x is a `tf.RaggedTensor` equivalent `KerasTensor`."""
    return (
        isinstance(x, tf.RaggedTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.RaggedTensorSpec)
    )


class RaggedStructure:
    """Just the row structure of a `tf.RaggedTensor`."""

    def __init__(self, row_splits):
        self._row_splits = row_splits

    @property
    def row_splits(self):
        return self._row_splits

    @memoized_property
    def total_size(self):
        return self.row_splits[-1]

    @memoized_property
    def row_starts(self) -> IntTensor:
        return self._row_splits[:-1]

    @memoized_property
    def row_ends(self) -> IntTensor:
        return self._row_splits[1:]

    @memoized_property
    def row_lengths(self) -> IntTensor:
        return self.row_ends - self.row_starts

    @memoized_property
    def value_rowids(self) -> IntTensor:
        return tf.ragged.row_splits_to_segment_ids(self.row_splits)

    @memoized_property
    def nrows(self) -> IntTensor:
        return tf.size(self.row_starts)

    def as_ragged(self, x, validate=True) -> tf.RaggedTensor:
        return layers.from_row_splits(x, self.row_splits, validate=validate)

    @staticmethod
    def from_ragged(rt: tf.RaggedTensor):
        if ragged_rank(rt) != 1:
            raise NotImplementedError("TODO")
        return RaggedStructure(layers.row_splits(rt))

    @staticmethod
    def from_dense(dense):
        bs, n = tf.unstack(tf.shape(dense)[:2])
        row_splits = tf.range(0, (bs + 1) * n, n, dtype=tf.int64)
        return RaggedStructure(row_splits)

    @staticmethod
    def from_tensor(x):
        if is_ragged(x):
            return RaggedStructure.from_ragged(x)
        return RaggedStructure.from_dense(x)

    @staticmethod
    def from_row_splits(row_splits):
        return RaggedStructure(row_splits)
