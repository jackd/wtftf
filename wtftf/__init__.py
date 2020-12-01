try:
    import tensorflow as tf

    if tf.version.VERSION < "2.3":
        print("Warning: wtftf only tested with tf >= 2.3.")
    del tf  # clean up workspace
except ImportError as e:
    raise ImportError("failed to import tensorflow") from e
from . import meta, ragged, sparse

__all__ = [
    "meta",
    "ragged",
    "sparse",
]
