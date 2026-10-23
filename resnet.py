"""ResNet baseline preserving the supplied reshape/permute and residual order.

Generator inputs are flat triples [delta_C, C, C_star] per pair. This
baseline reshapes the complete flat vector to (dimension // 2, 2), then
permutes to (2, dimension // 2); its convolution sequence length is TWO.
It does not use the Inception shared pair encoder or three-channel layout.
"""
from tensorflow.keras.layers import (
    Input, Reshape, Permute, Conv1D, BatchNormalization, Activation,
    Add, Flatten, Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def make_resnet(dimension=128, num_filters=32, num_outputs=1, d1=64,
                d2=64, ks=3, depth=5, reg_param=0.0001,
                final_activation='sigmoid'):
    """Build the supplied baseline, uncompiled, with unchanged defaults."""
    for name, value in dict(dimension=dimension, num_filters=num_filters,
                            num_outputs=num_outputs, d1=d1, d2=d2,
                            ks=ks, depth=depth).items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if dimension % 2:
        raise ValueError('dimension must be even for the original reshape')
    inp = Input(shape=(dimension,))
    rs = Reshape((dimension // 2, 2))(inp)
    perm = Permute((2, 1))(rs)
    shortcut = Conv1D(num_filters, 1, padding='same',
                      kernel_regularizer=l2(reg_param))(perm)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)
    for _ in range(depth):
        conv1 = Conv1D(num_filters, ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, ks, padding='same',
                       kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    x = Flatten()(shortcut)
    for width in (d1, d2):
        x = Dense(width, kernel_regularizer=l2(reg_param))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(x)
    return Model(inputs=inp, outputs=out)


def make_multipair_resnet(pairs=4, plain_bits=32, **kwargs):
    """Adapt the input size, without changing generator feature ordering."""
    for name, value in dict(pairs=pairs, plain_bits=plain_bits).items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    return make_resnet(dimension=3 * pairs * plain_bits, **kwargs)
