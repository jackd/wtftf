import tensorflow as tf

from wtftf.random import global_generator_context, uniform

rng = tf.random.get_global_generator()
rng.reset_from_seed(0)
u0 = uniform(())

rng.reset_from_seed(0)
with global_generator_context(tf.random.Generator.from_seed(1)):
    u1 = uniform(())
u2 = uniform(())

assert u0 == u2
assert u1 != u2
