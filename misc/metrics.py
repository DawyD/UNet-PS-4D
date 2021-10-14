from tensorflow.keras.metrics import Metric
import tensorflow as tf
import math


class AvgAngleMetric(Metric):
    def __init__(self, name='avg_angle_metrics', **kwargs):
        super(AvgAngleMetric, self).__init__(name=name, **kwargs)

        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1, 3))
        y_pred = tf.reshape(y_pred, (-1, 3))
        dot_prod = tf.abs(tf.einsum('ij,ij->i', y_pred, y_true))
        errors = 180 * tf.acos(tf.minimum(tf.constant(1.0, dtype=tf.float32), dot_prod)) / math.pi

        self.numerator.assign_add(tf.reduce_sum(errors))
        self.denominator.assign_add(tf.cast(tf.shape(errors)[0], tf.float32))

    def result(self):
        return self.numerator / self.denominator

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.numerator.assign(0.0)
        self.denominator.assign(0.0)


class GaussAvgAngleMetrics(AvgAngleMetric):
    def __init__(self, w=32, k_size=1, spherical=False, name='gauss_avg_angle_metrics', scale=None, **kwargs):
        super(GaussAvgAngleMetrics, self).__init__(name=name, **kwargs)

        self.w = w
        self.k_size = k_size
        self.ks_half = self.k_size // 2
        self.spherical = spherical

    def update_state(self, y_true, y_pred, sample_weight=None):
        mp = tf.nn.max_pool(y_pred, ksize=2*self.w+1, strides=1, padding="SAME", data_format="NHWC")
        max_out = tf.math.equal(mp, y_pred)[..., 0]
        w = tf.cast(self.w, tf.float32)
        a = tf.where(max_out)

        aid = tf.unique(a[:, 0])[1]
        aid = tf.math.segment_min(tf.range(tf.shape(aid)[0]), aid)
        a = tf.gather(a, aid)

        if self.k_size != 1:
            padded_outputs = tf.pad(y_pred[..., 0], ((0, 0), (self.ks_half, self.ks_half), (self.ks_half, self.ks_half)), mode="REFLECT")

            m1_flat = a[:, 2] + padded_outputs.shape[2] * a[:, 1]

            r = tf.range(0, self.k_size, dtype=tf.int64)
            offsets = r[:, None] * padded_outputs.shape[2] + r
            patch_indices = tf.reshape(offsets + m1_flat[:, None, None], (tf.shape(m1_flat)[0], -1))
            patches = tf.gather(tf.reshape(padded_outputs, (tf.shape(m1_flat)[0], -1)), patch_indices, batch_dims=-1)
            patches = tf.reshape(patches, (tf.shape(m1_flat)[0], self.k_size, self.k_size))

            b = self.centre_of_mass(patches)
            xy = b + tf.cast(a[:, 1:3] - self.ks_half, dtype=tf.float32)
        else:
            xy = tf.cast(tf.stack((a[:, 2], a[:, 1]), axis=-1), dtype=tf.float32)

        if self.spherical:
            azim = (xy[..., 0] / (self.w - 1)) * math.pi * 2
            elev = (xy[..., 1] / (self.w - 1)) * math.pi / 2
            cos_elev = tf.math.cos(elev)
            z = tf.math.sin(elev)
            x = cos_elev * tf.math.cos(azim)
            y = cos_elev * tf.math.sin(azim)
            xyz = tf.stack((x, y, z), axis=-1)
        else:
            xy = (xy / ((w - 1) / 2)) - 1
            z = tf.sqrt(1 - tf.minimum(xy[:, 0] ** 2 + xy[:, 1] ** 2, 1.))[..., None]
            xyz = tf.concat((xy, z), axis=1)

        super(GaussAvgAngleMetrics, self).update_state(y_true, xyz)

        return xyz

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {
            'w': self.w,
            'k_size': self.k_size,
            'spherical': self.spherical
        }
        base_config = super(GaussAvgAngleMetrics, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def centre_of_mass(volumes):
        shape = tf.shape(volumes)
        # Make array of coordinates (each row contains three coordinates)
        ii, jj = tf.meshgrid(tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
        coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
        coords = tf.cast(coords, tf.float32)
        # Rearrange input into one vector per volume
        volumes_flat = tf.reshape(volumes, [-1, shape[1] * shape[2], 1])
        # Compute total mass for each volume
        total_mass = tf.reduce_sum(volumes_flat, axis=1)
        # Compute centre of mass
        centre_of_mass = tf.reduce_sum(volumes_flat * coords, axis=1) / total_mass
        return centre_of_mass
