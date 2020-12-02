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

### Composite Tensors with keras layers

[This issue](https://github.com/tensorflow/tensorflow/issues/27170) has been open for over a year. TF 2.4 looks like it's going in the right direction. Until a stable 2.4 release, this repository aims to make building keras models with compound tensors easier.

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

A convenient way of wrapping a function with a `tf.keras.layers.Lambda` without losing the signature, name etc. This is mostly used for composite tensor ops discussed above, but can also be useful for wrapping `tf.py_function` or `tf.numpy_function` calls in `tf < 2.4`. See [examples/layer_wrapper.py](examples/layer_wrapper.py). For `tf >= 2.4` this does nothing except for classes (e.g. `tf.SparseTensor`).

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
