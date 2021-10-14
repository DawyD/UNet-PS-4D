from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Conv2DTranspose, Concatenate, Reshape, LeakyReLU
from tensorflow.keras.models import Model


def unet(input_shape, out_channels: int, nr_feats: int, nr_blocks: int, nr_conv: int, use_BN=True, use_bias=False):
    inp = Input(input_shape, name='input_image')
    net = inp

    levels = []
    for i in range(nr_blocks):
        for _ in range(nr_conv):
            net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias)(net)
            if use_BN:
                net = BatchNormalization(scale=False, center=False)(net)
            net = ReLU()(net)

        levels.append(net)
        net = MaxPool2D()(net)
        nr_feats = nr_feats * 2

    for _ in range(nr_conv):
        net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias)(net)
        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)

    for i in range(nr_blocks):
        nr_feats = nr_feats // 2
        net = Conv2DTranspose(nr_feats, 3, strides=2, padding="same", use_bias=use_bias)(net)

        if use_BN:
            net = BatchNormalization(scale=False, center=False)(net)
        net = ReLU()(net)
        net = Concatenate(axis=-1)([net, levels.pop()])

        for _ in range(nr_conv):
            net = Conv2D(nr_feats, 3, padding="same", use_bias=use_bias)(net)
            if use_BN:
                net = BatchNormalization(scale=False, center=False)(net)
            net = ReLU()(net)

    last = Conv2D(out_channels, 1, padding="same", use_bias=use_bias)(net)

    return Model(inputs=inp, outputs=last, name="UNet")

