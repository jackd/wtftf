import functools

import tensorflow as tf


@functools.wraps(tf.RaggedTensor.ragged_rank)
def ragged_rank(rt) -> int:
    """ragged_rank from `RaggedTensor` or `KerasTensor` equivalent."""
    if isinstance(rt, tf.RaggedTensor):
        return rt.ragged_rank
    else:
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
