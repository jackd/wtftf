# WTF Tensorflow (wtftf)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Various utilities that get around certain surprising / annoying "features" of [tensorflow](https://github.com/tensorflow/tensorflow.git).

## Installation

```bash
git clone https://github.com/jackd/wtftf.git
pip install -e wtftf
```

## Examples

See [examples](examples) for example usage.

### Random behaviour with `[get|set]_global_generator`

One might think that `tf.random.get_global_generator` would set the generator used by operations like `tf.random.uniform`. One would be wrong.

```python
import tensorflow as tf

import wtftf

tf.random.set_global_generator(tf.random.Generator.from_seed(0))
u0 = tf.random.uniform(())
tf.random.set_global_generator(tf.random.Generator.from_seed(0))
u1 = tf.random.uniform(())
assert u0 == u1  # fails
```

`wtftf.random` ops fix this by calling `tf.random.get_random_generator().random` (or other methods) under the hood.

```python
tf.random.set_global_generator(tf.random.Generator.from_seed(0))
u0 = wtftf.random.uniform(())
tf.random.set_global_generator(tf.random.Generator.from_seed(0))
u1 = wtftf.random.uniform(())
assert u0 == u1  # succeeds
```

### Composite Tensors with keras layers

[This issue](https://github.com/tensorflow/tensorflow/issues/27170) has been open for over a year and with tf 2.4 composite tensor support for building keras models with the functional API looks like it's going in the wrong direction.

Example that DOES NOT WORK

```python
import tensorflow as tf

values = tf.keras.Input((None,))
row_splits = tf.keras.Input((), dtype=tf.int32)
args = (values, row_splits)
tf.keras.Model(
    args,
    tf.RaggedTensor.from_row_splits(*args, validate=False),
)
```

Example that DOES WORK

```python
import tensorflow as tf
import wtftf.ragged.layers as ragged_layers

values = tf.keras.Input((None,))
row_splits = tf.keras.Input((), dtype=tf.int32)
args = (values, row_splits)
tf.keras.Model(args, ragged_layers.from_row_splits(*args))
```

### `meta.layered`

A convenient way of wrapping a function with a `tf.keras.layers.Lambda` without losing the signature, name etc. This is mostly used for composite tensor ops discussed above, but can also be useful for wrapping `tf.py_function` or `tf.numpy_function` calls in `tf < 2.4`. See [examples/layer_wrapper.py](examples/layer_wrapper.py).

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
