import tensorflow as tf


def is_sparse(x) -> bool:
    """Indicate if x is a `tf.SparseTensor` or equivalent `KerasTensor`."""
    return (
        isinstance(x, tf.SparseTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.SparseTensorSpec)
    )
