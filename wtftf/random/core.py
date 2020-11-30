import contextlib
from typing import Optional, Union

import tensorflow as tf

from wtftf.meta import as_static_method
from wtftf.random import rng as rng_lib


@contextlib.contextmanager
def global_generator_context(rng: Optional[tf.random.Generator] = None):
    if rng is None:
        rng = tf.random.Generator.from_non_deterministic_state()

    old_rng = tf.random.get_global_generator()
    try:
        tf.random.set_global_generator(rng)
        yield rng
    finally:
        tf.random.set_global_generator(old_rng)


def _export_generator_method(name: str):
    return as_static_method(tf.random.Generator, tf.random.get_global_generator, name)


binomial = _export_generator_method("binomial")
normal = _export_generator_method("normal")
truncated_normal = _export_generator_method("truncated_normal")
uniform = _export_generator_method("uniform")
uniform_full_int = _export_generator_method("uniform_full_int")
reset_from_seed = _export_generator_method("uniform_full_int")

# alias
set_seed = reset_from_seed


def perm(size: Union[int, tf.Tensor], name: Optional[str] = None):
    """Get a random permutation of the given `size`."""
    return rng_lib.perm(tf.random.get_global_generator(), size, name=name)


def shuffle(value, name: Optional[str] = None):
    """Shuffle `value` along leading dimension using global generator."""
    return rng_lib.shuffle(tf.random.get_global_generator(), value, name=name)
