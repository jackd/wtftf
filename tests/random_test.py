import numpy as np
import tensorflow as tf

import wtftf


class RandomTest(tf.test.TestCase):
    def test_uniform(self):
        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u0 = wtftf.random.uniform(())

        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u1 = wtftf.random.uniform(())

        self.assertEqual(u0, u1)

    def test_perm_deterministic(self):
        size = 1024
        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u0 = wtftf.random.perm(size)

        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u1 = wtftf.random.perm(size)

        np.testing.assert_equal(*self.evaluate((u0, u1)))

    def test_shuffle_deterministic(self):
        size = 1024
        values = tf.random.uniform((size,), dtype=tf.float32)
        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u0 = wtftf.random.shuffle(values)

        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u1 = wtftf.random.shuffle(values)

        np.testing.assert_equal(*self.evaluate((u0, u1)))

    def test_context(self):
        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u0 = wtftf.random.uniform(())
        u1 = wtftf.random.uniform(())

        tf.random.set_global_generator(tf.random.Generator.from_seed(0))
        u0_b = wtftf.random.uniform(())

        with wtftf.random.global_generator_context(tf.random.Generator.from_seed(1)):
            u_ctx = wtftf.random.uniform(())

        u1_b = wtftf.random.uniform(())

        tf.random.set_global_generator(tf.random.Generator.from_seed(1))
        u_ctx_b = wtftf.random.uniform(())

        self.assertEqual(u0, u0_b)
        self.assertEqual(u1, u1_b)
        self.assertEqual(u_ctx, u_ctx_b)

    def test_shuffle(self):
        x = tf.range(10)
        shuff = wtftf.random.shuffle(x)
        assert x.shape == (10,)
        np.testing.assert_equal(*self.evaluate((x, tf.sort(shuff))))


if __name__ == "__main__":
    tf.test.main()
