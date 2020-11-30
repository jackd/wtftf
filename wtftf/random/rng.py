""""Additional functions taking an explicit `Generator`."""
from typing import Optional, Union

import tensorflow as tf


def perm(
    rng: tf.random.Generator, size: Union[int, tf.Tensor], name: Optional[str] = None
):
    """Get a random permutation of size `size`."""
    with tf.name_scope(name or "shuffled_range"):
        i = rng.uniform((size,))
        return tf.argsort(i)


def shuffle(rng: tf.random.Generator, value, name: Optional[str] = None):
    """Shuffle `value` along leading axis."""
    if value.shape.ndims == 0:
        return value
    with tf.name_scope(name or "shuffle"):
        return tf.gather(value, perm(rng, tf.shape(value)[0]), axis=0)
