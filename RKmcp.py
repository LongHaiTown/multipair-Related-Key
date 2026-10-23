"""Multi-pair Inception-ECA model matching training_distinguisher_simplified.py.

Default input: flattened (batch, pairs * 3 * selected_bits) from
NDCMultiPairGenerator, whose per-pair order is [delta_C, C, C_prime].
Internally converted to (batch, pairs, selected_bits, 3), with channels
[C, C_prime, delta_C]. Set input_format="structured" for an already
structured tensor in that latter channel order.
Previously trained flat-input RKmcp.py weights are not compatible with this model.
"""

import math

from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Input, Multiply, Reshape,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def residual_block(x, filters, *, kernel_size=3, dilation_rate=1,
                   reg_param=1e-4, name):
    shortcut = x
    y = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               use_bias=False, kernel_regularizer=l2(reg_param),
               name=f"{name}_conv1")(x)
    y = BatchNormalization(name=f"{name}_bn1")(y)
    y = Activation("relu", name=f"{name}_relu1")(y)
    y = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
               use_bias=False, kernel_regularizer=l2(reg_param),
               name=f"{name}_conv2")(y)
    y = BatchNormalization(name=f"{name}_bn2")(y)
    x = Add(name=f"{name}_add")([shortcut, y])
    return Activation("relu", name=f"{name}_out")(x)


def eca_module(x, *, name):
    """Match the ECA kernel rule used by the training script exactly."""
    channels = int(x.shape[-1])
    kernel_size = int(abs(math.log2(channels) / 2 + 1))
    kernel_size += int(kernel_size % 2 == 0)
    weights = GlobalAveragePooling1D(name=f"{name}_gap")(x)
    weights = Reshape((channels, 1), name=f"{name}_reshape1")(weights)
    weights = Conv1D(1, kernel_size, padding="same", use_bias=False,
                     name=f"{name}_conv")(weights)
    weights = Activation("sigmoid", name=f"{name}_sigmoid")(weights)
    weights = Reshape((1, channels), name=f"{name}_reshape2")(weights)
    return Multiply(name=f"{name}_reweight")([x, weights])


def build_pair_encoder_inception(n_bits, *, branch_filters=32, pair_width=128,
                                 depth=3, embedding_dim=96, reg_param=1e-4,
                                 use_eca=True):
    inp = Input((n_bits, 3), name="pair_input")
    branches = [
        Conv1D(branch_filters, kernel, padding="same", use_bias=False,
               kernel_regularizer=l2(reg_param), name=f"pair_inception_k{kernel}")(inp)
        for kernel in (1, 3, 5, 7)
    ]
    x = Concatenate(name="pair_inception_concat")(branches)
    x = BatchNormalization(name="pair_inception_bn")(x)
    x = Activation("relu", name="pair_inception_relu")(x)
    if use_eca:
        x = eca_module(x, name="pair_eca_stem")
    x = Conv1D(pair_width, 1, padding="same", use_bias=False,
               kernel_regularizer=l2(reg_param), name="pair_projection")(x)
    x = BatchNormalization(name="pair_projection_bn")(x)
    x = Activation("relu", name="pair_projection_relu")(x)
    for i in range(depth):
        x = residual_block(x, pair_width, dilation_rate=(1, 2, 4)[i % 3],
                           reg_param=reg_param, name=f"pair_res_{i + 1}")
    if use_eca:
        x = eca_module(x, name="pair_eca_final")
    avg = GlobalAveragePooling1D(name="pair_gap")(x)
    maximum = GlobalMaxPooling1D(name="pair_gmp")(x)
    x = Concatenate(name="pair_pool_concat")([avg, maximum])
    x = Dense(embedding_dim, kernel_regularizer=l2(reg_param),
              name="pair_embedding_dense")(x)
    x = BatchNormalization(name="pair_embedding_bn")(x)
    out = Activation("relu", name="pair_embedding_relu")(x)
    return Model(inp, out, name="shared_inception_eca_pair_encoder")


def _make_model_inception(pairs, plain_bits, *, use_eca, pair_depth=3,
                          branch_filters=32, pair_width=128, pair_embedding_dim=96,
                          aggregator_filters=128, aggregator_depth=2, dense1=128,
                          dense2=64, dropout=0.20, reg_param=1e-4,
                          input_format="flat"):
    if not isinstance(pairs, int) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if not isinstance(plain_bits, int) or plain_bits < 1:
        raise ValueError("plain_bits (number of selected ciphertext bits) must be positive")
    if any(not isinstance(v, int) or v < 1 for v in (
        pair_depth, branch_filters, pair_width, pair_embedding_dim,
        aggregator_filters, dense1, dense2,
    )) or not isinstance(aggregator_depth, int) or aggregator_depth < 0:
        raise ValueError("depths and channel dimensions must be valid nonnegative integers")
    if not 0 <= dropout < 1 or reg_param < 0:
        raise ValueError("dropout must be in [0, 1) and reg_param must be nonnegative")
    if input_format not in ("flat", "structured"):
        raise ValueError("input_format must be 'flat' or 'structured'")

    encoder = build_pair_encoder_inception(
        plain_bits, branch_filters=branch_filters, pair_width=pair_width,
        depth=pair_depth, embedding_dim=pair_embedding_dim,
        reg_param=reg_param, use_eca=use_eca,
    )
    if input_format == "flat":
        inp = Input((pairs * 3 * plain_bits,), name="multi_pair_input")
        x = Reshape((pairs, 3, plain_bits), name="flat_pairs_to_words")(inp)
        x = keras.layers.Permute((1, 3, 2), name="words_to_bit_channels")(x)
        # Generator: [delta_C, C, C_prime]; encoder: [C, C_prime, delta_C].
        x = keras.ops.take(x, [1, 2, 0], axis=-1)
    else:
        inp = Input((pairs, plain_bits, 3), name="multi_pair_input")
        x = inp
    x = TimeDistributed(encoder, name="pair_encoder_td")(x)
    x = Conv1D(aggregator_filters, 1, padding="same", use_bias=False,
               kernel_regularizer=l2(reg_param), name="aggregator_projection")(x)
    x = BatchNormalization(name="aggregator_projection_bn")(x)
    x = Activation("relu", name="aggregator_projection_relu")(x)
    for i in range(aggregator_depth):
        x = residual_block(x, aggregator_filters, reg_param=reg_param,
                           name=f"agg_res_{i + 1}")
    avg = GlobalAveragePooling1D(name="multi_pair_avg_pool")(x)
    maximum = GlobalMaxPooling1D(name="multi_pair_max_pool")(x)
    x = Concatenate(name="multi_pair_pool_concat")([avg, maximum])
    x = Dense(dense1, kernel_regularizer=l2(reg_param), name="head_dense1")(x)
    x = BatchNormalization(name="head_bn1")(x)
    x = Activation("relu", name="head_relu1")(x)
    x = Dropout(dropout, name="head_dropout")(x)
    x = Dense(dense2, kernel_regularizer=l2(reg_param), name="head_dense2")(x)
    x = BatchNormalization(name="head_bn2")(x)
    x = Activation("relu", name="head_relu2")(x)
    out = Dense(1, activation="sigmoid", kernel_regularizer=l2(reg_param),
                name="output")(x)
    return Model(inp, out, name="MultiPair_Inception_ECA")


def make_model_inception(pairs=4, plain_bits=32, **kwargs):
    """Build the training-script Inception architecture for either input format.

    `plain_bits` means the number of selected bits, e.g. 30, 36, or 64.
    Flat input matches NDCMultiPairGenerator; structured input matches the
    separate training_distinguisher_simplified.py generator. Weights match
    across formats only when the data representation and hyperparameters match.
    """
    return _make_model_inception(pairs, plain_bits, use_eca=True, **kwargs)


def make_model_inception_no_eca(pairs=4, plain_bits=32, **kwargs):
    """Ablation with both pair-level ECA modules removed; retrain from scratch."""
    return _make_model_inception(pairs, plain_bits, use_eca=False, **kwargs)
