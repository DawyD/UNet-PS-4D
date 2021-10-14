from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape, Input, ReLU, Dropout, Concatenate, Flatten, Dense, Lambda, \
                                    AvgPool2D

from misc.layres import RotateVector, Random90Rotation
from models.utils_sep4D import obsConv2D, spatialConv2D, separableConv4D, obsPool, spatialPool


def densenet_separable4D(obs_shape, spatial_shape, nr_channels, rotate90=False):
    inputs1 = Input(spatial_shape + obs_shape + (nr_channels,))

    x0 = inputs1

    if rotate90:
        x0, k = Random90Rotation()(x0)
    else:
        k = None

    x1 = obsConv2D(x0, 16, (3, 3), name='conv1_obs', padding='same')
    x1 = spatialConv2D(x1, 16, (3, 3), name='conv1_spatial', padding='valid')

    # 1st Denseblock
    x1a = ReLU()(x1)

    x2 = separableConv4D(x1a, 16, name='conv2')
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = separableConv4D(xc1a, 16, name='conv3')
    x3 = Dropout(0.2)(x3)

    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    # Transition
    xc2a = ReLU()(xc2)
    x4 = obsConv2D(xc2a, 48, (1, 1), name='conv4_obs')
    x4 = Dropout(0.2)(x4)
    x1 = obsPool(x4)
    x1 = spatialPool(x1)

    # 2nd Dense block
    x1a = ReLU()(x1)
    x2 = separableConv4D(x1a, 16, name='conv5')
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = separableConv4D(xc1a, 16, name='conv6')
    x3 = Dropout(0.2)(x3)
    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    xc2a = ReLU()(xc2)
    x4 = spatialConv2D(xc2a, 80, (3, 3), name='conv7_spatial', padding='valid')
    x4 = obsConv2D(x4, 80, (1, 1), name='conv7_obs', padding='same')

    x = Flatten()(x4)
    x = Dense(128, activation='relu', name='dense1b')(x)
    x = Dense(3, name='dense2')(x)

    normalize = Lambda(lambda z: K.l2_normalize(z, axis=-1))
    x = normalize(x)

    if rotate90:
        x = RotateVector()(x, -k)

    outputs = x

    model = Model(inputs=inputs1, outputs=outputs)
    return model


def densent_separable4D_3x3(obs_shape, nr_channels, rotate90=False):
    inputs1 = Input((3, 3) + obs_shape + (nr_channels,))

    x0 = inputs1  # e.g. 32x32x5x5x3

    if rotate90:
        x0, k = Random90Rotation()(x0)
    else:
        k = None

    x1 = obsConv2D(x0, 16, (3, 3), name='conv1_obs', padding='same')
    x1 = spatialConv2D(x1, 16, (3, 3), name='conv1_spatial', padding='valid')
    x1 = Reshape(x1.shape[3:])(x1)

    # 1st Denseblock
    x1a = ReLU()(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv2')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=3)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv3')(xc1a)
    x3 = Dropout(0.2)(x3)

    xc2 = Concatenate(axis=3)([x3, x2, x1])

    # Transition
    xc2a = ReLU()(xc2)
    x4 = Conv2D(48, (1, 1), padding='same', name='conv4')(xc2a)
    x4 = Dropout(0.2)(x4)
    x1 = AvgPool2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x1a = ReLU()(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv5')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=3)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv6')(xc1a)
    x3 = Dropout(0.2)(x3)
    xc2 = Concatenate(axis=3)([x3, x2, x1])

    xc2a = ReLU()(xc2)
    x4 = Conv2D(80, (1, 1), padding='same', name='conv7')(xc2a)

    x = Flatten()(x4)
    x = Dense(128, activation='relu', name='dense1b')(x)
    x = Dense(3, name='dense2')(x)

    normalize = Lambda(lambda z: K.l2_normalize(z, axis=-1))
    x = normalize(x)

    if rotate90:
        x = RotateVector()(x, -k)

    outputs = x

    model = Model(inputs=inputs1, outputs=outputs)
    return model


def densent_separable4D_5x5(obs_shape, nr_channels, rotate90=False):
    inputs1 = Input((5, 5) + obs_shape + (nr_channels,))

    x0 = inputs1

    if rotate90:
        x0, k = Random90Rotation()(x0)
    else:
        k = None

    x1 = obsConv2D(x0, 16, (3, 3), name='conv1_obs', padding='valid')
    x1 = spatialConv2D(x1, 16, (3, 3), name='conv1_spatial', padding='valid')

    # 1st Denseblock
    x1a = ReLU()(x1)
    x2 = obsConv2D(x1a, 16, (3, 3), name='conv2_obs', padding='same')
    x2 = spatialConv2D(x2, 16, (3, 3), name='conv2_spatial', padding='same')
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = obsConv2D(xc1a, 16, (3, 3), name='conv3_obs', padding='same')
    x3 = spatialConv2D(x3, 16, (3, 3), name='conv3_spatial', padding='same')
    x3 = Dropout(0.2)(x3)

    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    # Transition
    xc2a = ReLU()(xc2)
    x4 = obsConv2D(xc2a, 16, (3, 3), name='conv4_obs', padding='valid')
    x4 = spatialConv2D(x4, 16, (3, 3), name='conv4_spatial', padding='valid')
    x4 = Reshape(x4.shape[3:])(x4)
    x4 = Dropout(0.2)(x4)
    x1 = AvgPool2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x1a = ReLU()(x1)
    x2 = Conv2D(32, (3, 3), padding='same', name='conv5')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = Conv2D(32, (3, 3), padding='same', name='conv6')(xc1a)
    x3 = Dropout(0.2)(x3)
    xc2 = Concatenate(axis=-1)([x3, x2, x1])

    xc2a = ReLU()(xc2)
    x4 = Conv2D(32, (3, 3), padding='valid', name='conv7')(xc2a)
    x4 = Dropout(0.2)(x4)
    x1 = AvgPool2D((2, 2), strides=(2, 2))(x4)

    # 3nrd Dense block
    x1a = ReLU()(x1)
    x2 = Conv2D(64, (3, 3), padding='same', name='conv8')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = Concatenate(axis=-1)([x2, x1])

    xc1a = ReLU()(xc1)
    x3 = Conv2D(64, (3, 3), padding='same', name='conv9')(xc1a)
    x3 = Dropout(0.2)(x3)
    x = Concatenate(axis=-1)([x3, x2, x1])

    x = ReLU()(x)

    x = Conv2D(128, (3, 3), padding='valid', name='conv10')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', name='dense1b')(x)
    x = Dense(3, name='dense2')(x)

    normalize = Lambda(lambda z: K.l2_normalize(z, axis=-1))
    x = normalize(x)

    if rotate90:
        x = RotateVector()(x, -k)

    outputs = x

    model = Model(inputs=inputs1, outputs=outputs)
    return model
