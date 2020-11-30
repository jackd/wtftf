import tensorflow as tf

import wtftf.ragged.layers as ragged_layers


class RaggedLayersTest(tf.test.TestCase):
    if tf.version.VERSION < "2.4":

        def test_tf_model_construction_raises(self):
            with self.assertRaises(ValueError):
                values = tf.keras.Input((None,))
                row_splits = tf.keras.Input((), dtype=tf.int32)
                args = (values, row_splits)
                tf.keras.Model(
                    args, tf.RaggedTensor.from_row_splits(*args, validate=False),
                )

    def test_wtftf_model_construction(self):
        del self
        values = tf.keras.Input((None,))
        row_splits = tf.keras.Input((), dtype=tf.int32)
        args = (values, row_splits)
        tf.keras.Model(args, ragged_layers.from_row_splits(*args))


if __name__ == "__main__":
    tf.test.main()
    # RaggedLayersTest().test_tf_model_construction_raises()
