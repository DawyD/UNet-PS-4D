import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss, Reduction, mean_squared_error
from misc.layres import Gauss2D


class SpatialGaussMSE2(Loss):
    def __init__(self, scale, w, reduction=Reduction.AUTO, name='spatial_gauss_mse'):
        super().__init__(reduction=reduction, name=name)
        self.scale = scale
        self.w = w
        self.gauss_layer = Gauss2D(w=self.w, scale=self.scale)

    def call(self, y_true, y_pred):
        y_true = K.reshape(y_true, (-1, 3))
        y_pred = K.reshape(y_pred, (-1, self.w, self.w, 1))
        loss = mean_squared_error(self.gauss_layer(y_true) * 100, y_pred)
        return loss

    def get_config(self):
        config = {
            'scale': self.scale,
            'w': self.w
        }
        base_config = super(SpatialGaussMSE2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussMSE(Loss):
    def __init__(self, scale, w, reduction=Reduction.AUTO, name='spatial_gauss_mse'):
        super().__init__(reduction=reduction, name=name)
        self.scale = scale
        self.w = w
        self.gauss_layer = Gauss2D(w=self.w, scale=self.scale)

    def call(self, y_true, y_pred):
        #if len(y_true.shape) == 4:
        #    y_true = tf.reshape(y_true, [-1] + list(y_true.shape[3:]))
        #    y_pred = tf.reshape(y_pred, [-1] + list(y_pred.shape[3:]))
        loss = mean_squared_error(self.gauss_layer(y_true) * 100, y_pred)
        return loss

    def get_config(self):
        config = {
            'scale': self.scale,
            'w': self.w
        }
        base_config = super(GaussMSE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProjectedSoftmax2D(Loss):
    def __init__(self, w, reduction=Reduction.AUTO, name='spatial_gauss_mse'):
        super().__init__(reduction=reduction, name=name)
        self.w = w

    def call(self, y_true, y_pred):
        m = K.cast(K.round((y_true + 1) * (self.w - 1) / 2), tf.int32)
        m = m[:, 1] * self.w + m[:, 0]
        pred = K.reshape(y_pred, (-1, self.w * self.w))
        loss = tf.keras.losses.sparse_categorical_crossentropy(m, pred, from_logits=True, axis=-1)
        return loss

    def get_config(self):
        config = {
            'w': self.w
        }
        base_config = super(ProjectedSoftmax2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
