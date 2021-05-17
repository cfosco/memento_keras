# Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

"""Modulus model templates for ResNets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model

from modulus.decorators.arg_scope import arg_scope
from modulus.models.templates.utils import add_dense_head, CNNBlock, get_batchnorm_axis


def ResNet(nlayers, inputs, use_batch_norm=False, data_format=None, add_head=False,
           nclasses=None, kernel_regularizer=None, bias_regularizer=None, activation_type='relu',
           all_projections=False, use_pooling=True):
    """
    Construct a fixed-depth vanilla ResNet, based on the architectures from the original paper [1].

    Args:
        nlayers (int): the number of layers in the desired ResNet (e.g. 18, 34, ..., 152).
        inputs (tensor): the input tensor.
        use_batch_norm (bool): whether batchnorm should be added after each convolution.
        data_format (str): either 'channels_last' (NHWC) or 'channels_first' (NCHW).
        add_head (bool): whether to add the original [1] classification head. Note that if you
            don't include the head, the actual number of layers in the model produced by this
            function is 'nlayers-1`.
        nclasses (int): the number of classes to be added to the classification head. Can be `None`
            if unused.
        kernel_regularizer: regularizer to apply to kernels.
        bias_regularizer: regularizer to apply to biases.
        all_projections (bool): whether to implement cnn subblocks with all shortcuts connections
            forced as 1x1 convolutional layers as mentioned in [1] to enable full pruning of
            ResNets. If set as False, the template instantiated will be the classic ResNet template
            as in [1] with shortcut connections as skip connections when there is no stride change
            and 1x1 convolutional layers (projection layers) when there is a stride change.
            Note: The classic template cannot be fully pruned. Only the first N-1 number of layers
            in the ResNet subblock can be pruned. All other layers must be added to exclude layers
            list while pruning, including conv1 layer.
        use_pooling (bool): whether to use MaxPooling2D layer after first conv layer or use a
        stride of 2 for first convolutional layer in subblock
    Returns:
        Model: the output model after applying the ResNet on top of input `x`.

    [1] Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    """
    if data_format is None:
        data_format = K.image_data_format()

    if add_head is False and nclasses is not None:
        raise ValueError('nclasses is defined (as %s) while add_head is `False`.' % nclasses)

    x = Conv2D(64,
               (7, 7),
               strides=(2, 2),
               padding='same',
               data_format=data_format,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               name='conv1')(inputs)
    if use_batch_norm:
        x = BatchNormalization(axis=get_batchnorm_axis(data_format), name='bn_conv1')(x)
    x = Activation(activation_type)(x)
    first_stride = 2  # Setting stride 1st convolutional subblock.
    last_stride = 1  # Setting stride last convolutional subblock.
    if use_pooling:
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                         data_format=data_format)(x)
        first_stride = 1
        last_stride = 2

    # Define a block functor which can create blocks.
    with arg_scope([CNNBlock],
                   use_batch_norm=use_batch_norm,
                   all_projections=all_projections,
                   use_shortcuts=True,
                   data_format=data_format,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   activation_type=activation_type):
        if nlayers == 10:
            x = CNNBlock(repeat=1, stride=first_stride, subblocks=[(3, 64), (3, 64)], index=1)(x)
            x = CNNBlock(repeat=1, stride=2, subblocks=[(3, 128), (3, 128)], index=2)(x)
            x = CNNBlock(repeat=1, stride=2, subblocks=[(3, 256), (3, 256)], index=3)(x)
            x = CNNBlock(repeat=1, stride=last_stride, subblocks=[(3, 512), (3, 512)], index=4)(x)
        elif nlayers == 18:
            x = CNNBlock(repeat=2, stride=first_stride, subblocks=[(3, 64), (3, 64)], index=1)(x)
            x = CNNBlock(repeat=2, stride=2, subblocks=[(3, 128), (3, 128)], index=2)(x)
            x = CNNBlock(repeat=2, stride=2, subblocks=[(3, 256), (3, 256)], index=3)(x)
            x = CNNBlock(repeat=2, stride=last_stride, subblocks=[(3, 512), (3, 512)], index=4)(x)
        elif nlayers == 34:
            x = CNNBlock(repeat=3, stride=first_stride, subblocks=[(3, 64), (3, 64)], index=1)(x)
            x = CNNBlock(repeat=4, stride=2, subblocks=[(3, 128), (3, 128)], index=2)(x)
            x = CNNBlock(repeat=6, stride=2, subblocks=[(3, 256), (3, 256)], index=3)(x)
            x = CNNBlock(repeat=3, stride=last_stride, subblocks=[(3, 512), (3, 512)], index=4)(x)
        elif nlayers == 50:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)], index=1)(x)
            x = CNNBlock(repeat=4, stride=2, subblocks=[(1, 128), (3, 128), (1, 512)], index=2)(x)
            x = CNNBlock(repeat=6, stride=2, subblocks=[(1, 256), (3, 256), (1, 1024)], index=3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
        elif nlayers == 101:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)], index=1)(x)
            x = CNNBlock(repeat=4, stride=2, subblocks=[(1, 128), (3, 128), (1, 512)], index=2)(x)
            x = CNNBlock(repeat=23, stride=2, subblocks=[(1, 256), (3, 256), (1, 1024)], index=3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
        elif nlayers == 152:
            x = CNNBlock(repeat=3, stride=first_stride,
                         subblocks=[(1, 64), (3, 64), (1, 256)], index=1)(x)
            x = CNNBlock(repeat=8, stride=2, subblocks=[(1, 128), (3, 128), (1, 512)], index=2)(x)
            x = CNNBlock(repeat=36, stride=2, subblocks=[(1, 256), (3, 256), (1, 1024)], index=3)(x)
            x = CNNBlock(repeat=3, stride=last_stride,
                         subblocks=[(1, 512), (3, 512), (1, 2048)], index=4)(x)
        else:
            raise NotImplementedError('A resnet with nlayers=%d is not implemented.' % nlayers)

    # Add AveragePooling2D layer if use_pooling is enabled after resnet block.
    if use_pooling:
        x = AveragePooling2D(pool_size=(7, 7), data_format=data_format, padding='same')(x)

    # Naming model.
    model_name = 'resnet%d' % nlayers
    if not use_pooling:
        model_name += '_nopool'
    if use_batch_norm:
        model_name += '_bn'
    # Set up keras model object.
    model = Model(inputs=inputs, outputs=x, name=model_name)

    # Add a dense head of nclasses if enabled.
    if add_head:
        model = add_dense_head(model,
                               inputs,
                               nclasses)
    return model