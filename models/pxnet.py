"""
PX-NET code based on the pre-print article https://arxiv.org/abs/2008.04933 (v1 and v2)
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, Input, Concatenate, Conv2D, MaxPool2D, AvgPool2D


def denseblock(x, channels):
    x1 = Conv2D(channels, (3, 3), padding='same', activation='relu')(x)
    x2 = Conv2D(channels, (3, 3), padding='same', activation='relu')(x1)
    xc2 = Concatenate(axis=-1)([x1, x2])
    x3 = Conv2D(channels, (3, 3), padding='same', activation='relu')(xc2)
    return Concatenate(axis=-1)([x, x1, x2, x3])


def transition(x, channels):
    x = Conv2D(channels, (1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    return Dropout(0.2)(x)


def pxnet_2D_v2(rows, cols, channels):
    inputs1 = Input((rows, cols, channels))

    x = Conv2D(32, (3, 3), padding='same', name='conv1', activation='relu')(inputs1)
    x = denseblock(x, 32)
    x = transition(x, 64)
    x = denseblock(x, 64)
    x = transition(x, 128)
    x = denseblock(x, 128)
    x = transition(x, 256)
    x = denseblock(x, 256)
    x = Flatten()(x)
    x = Dense(3)(x)

    normalize = Lambda(lambda z: K.l2_normalize(z, axis=-1))
    outputs = normalize(x)

    model = Model(inputs=inputs1, outputs=outputs)
    return model


def pxnet_2D_v1(rows, cols, channels):
    inputs1 = Input((rows, cols, channels))

    x1 = Conv2D(32, (3, 3), padding='same', name='conv1', activation='relu')(inputs1)

    # 1st Denseblock
    x2 = Conv2D(32, (3, 3), padding='same', name='conv2', activation='relu')(x1)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    x3 = Conv2D(32, (3, 3), padding='same', name='conv3', activation='relu')(xc1)
    x3 = Dropout(0.2)(x3)

    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    # Transition
    x4 = Conv2D(64, (1, 1), padding='same', name='conv4', activation='relu')(xc2)
    x4 = Dropout(0.2)(x4)
    x1 = AvgPool2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x2 = Conv2D(64, (3, 3), padding='same', name='conv5', activation='relu')(x1)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    x3 = Conv2D(64, (3, 3), padding='same', name='conv6', activation='relu')(xc1)
    x3 = Dropout(0.2)(x3)
    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    x4 = Conv2D(128, (3, 3), padding='same', name='conv7')(xc2)

    x = Flatten()(x4)
    x = Dense(128, activation='relu', name='dense1b')(x)
    x1 = Dense(64, activation='relu', name='dense2')(x)
    log = Lambda(lambda z: K.log(z))
    x2 = log(x)
    x = Concatenate(axis=-1)([x2, x1])
    x = Dense(3, name='dense3')(x)

    normalize = Lambda(lambda z: K.l2_normalize(z, axis=-1))
    x = normalize(x)

    outputs = x

    model = Model(inputs=inputs1, outputs=outputs)
    return model
