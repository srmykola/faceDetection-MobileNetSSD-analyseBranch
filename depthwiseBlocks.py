from __future__ import print_function

import numpy as np
import warnings

from keras_layer_AnchorBoxes import AnchorBoxes
from keras import backend as K
from keras import utils
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.advanced_activations import *
from keras.layers.pooling import *
from keras.activations import *
from keras.layers.convolutional import *
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.constraints import *
from keras.layers.noise import *

import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, ELU, Reshape, Concatenate, Activation

from keras_layer_L2Normalization import L2Normalization

mobilenet = True
separable_filter = False
conv_model = False
# tf = K.tf
dropout_rate = 0.55
W_regularizer = None
init_ = 'glorot_uniform'
conv_has_bias = True #False for BN
fc_has_bias = True


def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)



def _depthwise_conv_block_detection(input, layer_name, strides = (1,1),
                          kernel_size = 3,
                          pointwise_conv_filters=32, alpha=1.0, depth_multiplier=1,
                          padding = 'valid',
                          data_format = None,
                          activation = None, use_bias = True,
                          depthwise_initializer='glorot_uniform',
                          pointwise_initializer='glorot_uniform', bias_initializer = "zeros",
                          bias_regularizer= None, activity_regularizer = None,
                          depthwise_constraint = None, pointwise_constraint = None,
                          bias_constraint= None, batch_size = None,
                          block_id=1,trainable = None, weights = None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((kernel_size, kernel_size),
                        padding=padding,
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name=layer_name + '_conv_dw_%d' % block_id)(input)
    x = BatchNormalization(axis=channel_axis, name=layer_name + '_conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name=layer_name+'_conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               #padding='same',
               padding=padding,
               use_bias=False,
               strides=(1, 1),
               name=layer_name + '_conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,  name=layer_name+'_conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6,  name=layer_name+ '_conv_pw_%d_relu' % block_id)(x)
