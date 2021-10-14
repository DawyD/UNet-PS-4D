import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import math


class Random90Rotation(Layer):
    def __init__(self, **kwargs):
        super(Random90Rotation, self).__init__(**kwargs)

    @tf.function
    def call(self, x, training=None):
        # x: [batch, spatial_y, spatial_x, w, w, nr_channels]
        if training is None:
            training = K.learning_phase()

        if training:
            print("Training with random rotations")

            k = tf.random.uniform(minval=0, maxval=4, dtype=tf.int32, shape=())

            original_shape = list(x.shape)
            x = tf.reshape(x, [-1] + original_shape[1:3] + [int(np.prod(original_shape[3:]))])
            x = tf.image.rot90(x, k=k)  # rotate the spatial dimensions counter-clockwise
            x = tf.reshape(x, [-1] + original_shape[3:])
            x = tf.image.rot90(x, k=-k)  # rotate the obs. dims clockwise (counter-clockwise given the inverse y axis)
            x = tf.reshape(x, [-1] + original_shape[1:])

            return x, k
        else:
            print("Testing without random rotations")
            k = tf.constant(0, dtype=tf.int32, shape=())
            return x, k


class RotateVector(Layer):
    def __init__(self, **kwargs):
        super(RotateVector, self).__init__(**kwargs)

    @tf.function
    def call(self, x, k):
        theta = tf.cast(k, tf.float32) * tf.constant(math.pi / 2, dtype=tf.float32, shape=())
        matrix = tf.stack([tf.stack([tf.cos(theta), tf.sin(theta)], axis=0),
                           tf.stack([-tf.sin(theta),  tf.cos(theta)], axis=0)], axis=0)
        x12 = tf.tensordot(x[..., :2], matrix, axes=[[1], [0]])
        x = tf.concat([x12, x[..., 2:3]], axis=1)
        return x


class Gauss2D(Layer):
    def __init__(self, w, scale=2, **kwargs):
        super(Gauss2D, self).__init__(**kwargs)
        self.scale = scale
        self.w = w

    def call(self, nml):
        return tf.vectorized_map(self.fnc, nml[:, 1::-1])
        # return tf.map_fn(self.fnc, nml[:, :2])  # TODO: old and wrong version

    def fnc(self, x):
        m = (x + 1) * (self.w - 1) / 2
        d = tfp.distributions.MultivariateNormalDiag(loc=m, scale_diag=[self.scale, self.scale])
        idx = tf.range(start=0, limit=self.w*self.w, dtype=tf.int64)
        xy = tf.transpose(tf.unravel_index(indices=idx, dims=[self.w, self.w]))
        vals = d.prob(tf.cast(xy, tf.float32))
        maps = tf.reshape(vals, [self.w, self.w, 1])

        return maps

    def get_config(self):
        config = {
            'w': self.w,
            'scale': self.scale,
        }
        base_config = super(Gauss2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

