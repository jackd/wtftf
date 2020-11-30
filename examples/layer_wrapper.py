import numpy as np
import tensorflow as tf

from wtftf.meta import layered


def f_np(x, y):
    return np.sin(x) + y


def f_graph(x, y):
    return tf.numpy_function(f_np, (x, y), tf.float64)


f_layered = layered(f_graph)

x = tf.keras.Input(())
y = tf.keras.Input(())

try:
    out = f_graph(x, y)
    no_wrapper_success = True
    print("Succeeds without wrapper")
except NotImplementedError:
    no_wrapper_success = False
    #  Cannot convert a symbolic Tensor (input_1:0) to a numpy array.
    print("Fails without wrapper")


try:
    out = f_layered(x, y)
    wrapper_success = True
except NotImplementedError:
    wrapper_success = False


print(f"Without wrapper: {'success' if no_wrapper_success else 'failed'}")
print(f"With wrapper   : {'success' if wrapper_success else 'failed'}")
# both succeed with tf >= 2.4
# only wrapped succeeds with tf < 2.4
