"""
CNN-PS original network https://github.com/satoshi-ikehata/CNN-PS/
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, ReLU, Flatten, Lambda, Input, Concatenate, Conv2D, AvgPool2D


def densenet_2D(rows, cols, channels):

    inputs1 = Input((rows, cols, channels))

    x0 = inputs1
    x1 = Conv2D(16, (3, 3), padding='same', name='conv1')(x0)

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

    outputs = x

    model = Model(inputs=inputs1, outputs=outputs)
    return model


