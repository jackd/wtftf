import tensorflow as tf

import wtftf.sparse.layers as sparse_layers


class SparseLayersTest(tf.test.TestCase):
    if tf.version.VERSION < "2.4":

        def test_tf_model_construction_raises(self):
            with self.assertRaises(ValueError):
                indices = tf.keras.Input((2,), dtype=tf.int64)
                values = tf.keras.Input((), dtype=tf.float32)
                args = (indices, values)
                tf.keras.Model(
                    args, tf.SparseTensor(*args, dense_shape=(5, 5)),
                )

    def test_wtftf_model_construction(self):
        del self
        indices = tf.keras.Input((2,), dtype=tf.int64)
        values = tf.keras.Input((), dtype=tf.float32)
        args = (indices, values)
        tf.keras.Model(args, sparse_layers.SparseTensor(*args, dense_shape=(5, 5)))


if __name__ == "__main__":
    tf.test.main()
    # SparseLayersTest().test_tf_model_construction_raises()
