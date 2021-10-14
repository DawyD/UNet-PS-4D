import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, AvgPool2D, MaxPool2D


def grouped_conv2d(inputs, filters, kernel_size, strides=(1, 1), name=None, padding='same', axis=1, use_bias=True, transpose=False, **kwargs):
    shape = list(inputs.shape)
    if axis != 1:
        # [b,y,x,K,ch] -> [b,K,y,x,ch]
        inputs = tf.transpose(inputs, [0, axis] + list(range(1, axis)) + list(range(axis+1, len(shape))))
    inputs = tf.reshape(inputs, [-1] + list(inputs.shape[2:]))
    if not transpose:
        inputs = Conv2D(filters, kernel_size, strides, padding, name=name, use_bias=use_bias, **kwargs)(inputs)
    else:
        inputs = Conv2DTranspose(filters, kernel_size, strides, padding, name=name, use_bias=use_bias, **kwargs)(inputs)
    inputs = tf.reshape(inputs, [-1, shape[axis]] + list(inputs.shape[1:]))
    if axis != 1:
        # [b,K,y,x,ch] -> [b,y,x,K,ch]
        inputs = tf.transpose(inputs, [0] + list(range(2, axis+1)) + [1] + list(range(axis+1, len(shape))))
    return inputs


def obsConv2D(x, nr_feats, kernel_size, strides=(1, 1), name=None, padding='same', use_bias=True, transpose=False, **kwargs):
    # x: [batch, spatial_y, spatial_x, w, w, nr_channels]
    group_shape = x.shape[1:3]
    nr_group_channels = group_shape[0] * group_shape[1]
    x = Reshape((nr_group_channels,) + x.shape[3:])(x)
    x = grouped_conv2d(x, nr_feats, kernel_size, strides, name=name, padding=padding, axis=1, use_bias=use_bias, transpose=transpose, **kwargs)
    x = Reshape((group_shape + x.shape[2:]))(x)
    return x


def spatialConv2D(x, nr_feats, kernel_size, strides=(1, 1), name=None, padding='same', use_bias=True, transpose=False, **kwargs):
    # x: [batch, spatial_y, spatial_x, w, w, nr_channels]
    group_shape = x.shape[3:5]
    nr_group_channels = group_shape[0] * group_shape[1]
    x = Reshape(x.shape[1:3] + (nr_group_channels, x.shape[5],))(x)
    x = grouped_conv2d(x, nr_feats, kernel_size, strides, name=name, padding=padding, axis=3, use_bias=use_bias, transpose=transpose, **kwargs)
    x = Reshape((x.shape[1:3] + group_shape + (nr_feats,)))(x)
    return x


def spatialPool(x, reduction="average", **kwargs):
    # observation map pooling
    # x: [batch, spatial_y, spatial_x, w, w, nr_channels]
    group_shape = x.shape[3:5]
    nr_channels = x.shape[5]
    x = Reshape(x.shape[1:3] + (group_shape[0] * group_shape[1] * nr_channels,))(x)
    if reduction == "average":
        x = AvgPool2D(**kwargs)(x)
    elif reduction == "max":
        x = MaxPool2D(**kwargs)(x)
    else:
        raise ValueError("Unknown pooling layer")
    x = Reshape(x.shape[1:3] + (group_shape[0], group_shape[1], nr_channels))(x)
    return x


def obsPool(x, reduction="average", **kwargs):
    # spatial map pooling
    # x: [batch, spatial_y, spatial_x, w, w, nr_channels]
    group_shape = list(x.shape[1:3])
    x = tf.reshape(x, [-1] + list(x.shape[3:]))
    if reduction == "average":
        x = AvgPool2D(**kwargs)(x)
    elif reduction == "max":
        x = MaxPool2D(**kwargs)(x)
    else:
        raise ValueError("Unknown pooling layer")
    x = tf.reshape(x, [-1] + group_shape + list(x.shape[1:]))
    return x


def separableConv4D(x, nr_feats_out, obs_kernel_size=(3, 3), spatial_kernel_size=(3, 3), padding="same", use_bias=True, name=None):
    x = obsConv2D(x, nr_feats_out, obs_kernel_size, (1, 1), name + '_obs', padding=padding, use_bias=use_bias)
    x = spatialConv2D(x, nr_feats_out, spatial_kernel_size, (1, 1), name + '_spatial', padding=padding, use_bias=use_bias)
    return x


def separableConv4Dtransposed(x, nr_feats_out, obs_kernel_size=(3, 3), spatial_kernel_size=(3, 3), strides=(1,1), padding="same", use_bias=True, name=None):
    x = obsConv2D(x, nr_feats_out, obs_kernel_size, strides, name + '_obs', padding=padding, use_bias=use_bias, transpose=True)
    x = spatialConv2D(x, nr_feats_out, spatial_kernel_size, strides, name + '_spatial', padding=padding, use_bias=use_bias, transpose=True)
    return x


def separableAvgPool4D(x):
    x = obsPool(x)
    x = spatialPool(x, pool_size=(3, 3), strides=(1, 1))
    return x


def separableMaxPool4D(x):
    x = obsPool(x, reduction="max")
    x = spatialPool(x, reduction="max")
    return x