from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape, Input, ReLU, Concatenate, MaxPool2D, Conv2DTranspose, \
                                    BatchNormalization

from models.utils_sep4D import obsConv2D, spatialConv2D


def unet_sep4d(obs_shape, out_channels: int, nr_feats: int, nr_blocks: int, nr_conv: int, extra_up_blocks: int = 0,
               use_BN=True, use_bias=False, initializer="glorot_uniform"):
    if nr_blocks < 1:
        raise ValueError("Nr blocks has to be larger than 1 to use this model")
    if obs_shape[0] != 3 and obs_shape[0] != 5 and obs_shape[0] != 7 and obs_shape[0] != 9:
        raise NotImplementedError("Only (3,3), (5,5), and (7,7) spatial shapes are allowed for the moment")
    assert obs_shape[0] == obs_shape[1]

    inp = Input(obs_shape)
    net = inp

    levels = []

    if obs_shape[0] == 3:
        net = obsConv2D(net, nr_feats, (3, 3), name='conv1_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (3, 3), name='conv1_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)

        net = Reshape(net.shape[3:])(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias, kernel_initializer=initializer)(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    elif obs_shape[0] == 5:
        net = obsConv2D(net, nr_feats, (3, 3), name='conv1_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (3, 3), name='conv1_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = obsConv2D(net, nr_feats, (3, 3), name='conv2_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (3, 3), name='conv2_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        net = Reshape(net.shape[3:])(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    elif obs_shape[0] == 7:
        net = obsConv2D(net, nr_feats, (3, 3), name='conv1_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (5, 5), name='conv1_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = obsConv2D(net, nr_feats, (3, 3), name='conv2_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (3, 3), name='conv2_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        net = Reshape(net.shape[3:])(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    elif obs_shape[0] == 9:
        net = obsConv2D(net, nr_feats, (3, 3), name='conv1_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (5, 5), name='conv1_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = obsConv2D(net, nr_feats, (3, 3), name='conv2_obs', padding='same', use_bias=use_bias,
                        kernel_initializer=initializer)
        net = spatialConv2D(net, nr_feats, (5, 5), name='conv2_spatial', padding='valid', use_bias=use_bias,
                            kernel_initializer=initializer)
        net = Reshape(net.shape[3:])(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    for i in range(1, nr_blocks):
        for j in range(nr_conv):
            net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias, kernel_initializer=initializer)(net)
            if use_BN:
                net = BatchNormalization(scale=False, center=False)(net)
            net = ReLU()(net)

        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    for j in range(nr_conv):
        net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias, kernel_initializer=initializer)(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)

    for i in range(nr_blocks):
        nr_feats = nr_feats // 2
        net = Conv2DTranspose(nr_feats, 3, strides=2, padding="same", use_bias=use_bias,
                              kernel_initializer=initializer)(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = Concatenate(axis=-1)([net, levels.pop()])

        for j in range(nr_conv):
            net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias, kernel_initializer=initializer)(net)
            if use_BN:
                net = BatchNormalization(scale=False, center=False)(net)
            net = ReLU()(net)

    for i in range(extra_up_blocks):
        nr_feats = nr_feats // 2
        net = Conv2DTranspose(nr_feats, 3, strides=2, padding="same", use_bias=use_bias,
                              kernel_initializer=initializer)(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        for j in range(nr_conv):
            net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias, name="conv_ex%d_%d" % (i, j),
                         kernel_initializer=initializer)(net)
            net = ReLU()(net)

    last = Conv2D(out_channels, 1, padding="same", use_bias=False, kernel_initializer=initializer)(net)

    return Model(inputs=inp, outputs=last, name="UNet")
