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

mobilenet = True
separable_filter = False
conv_model = False
dropout_rate = 0.55
W_regularizer = None
init_ = 'glorot_uniform'
conv_has_bias = True #False for BN
fc_has_bias = True

#######################################################################
#Separable filter
def separable_res_block1(input_layer, layer_name, nb_filter, nb_row, subsample=(1,1)):
    x = input_layer
    x_right = bnConv_layer(x, layer_name + '_right', nb_filter, 1, nb_row, subsample)
    x_left  = bnConv_layer(x, layer_name + '_left',  nb_filter, nb_row, 1, subsample)
    x = Lambda(lambda z : (z[0] + z[1])/2., name = layer_name + '_merge')([x_right,x_left])
    return x


#######################################################################

#Depthwise convolution

def relu6(x):
    return K.relu(x, max_value=6)

class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depthMultiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depthMultiplier = depthMultiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3 #tf
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depthMultiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depthMultiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depthMultiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depthMultiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depthMultiplier'] = self.depthMultiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

########################################################################

def convBlock( inputs, filters, alpha, kernel = (3, 3), strides = (1, 1) ):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def bnConv(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample =(1,1), border_mode ='same', bias=conv_has_bias):
    tmp_layer = input_layer
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, activation=None, border_mode=border_mode, name = layer_name, bias=bias, init=init_, W_regularizer= W_regularizer)(tmp_layer)
    tmp_layer = BatchNormalization(name = layer_name + '_bn')(tmp_layer)
    tmp_layer = Lambda(lambda x:tf.nn.relu(x), name = layer_name + '_nonlin')(tmp_layer)
    return tmp_layer


def bnConv_layer(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample=(1,1), border_mode = 'same',bias=conv_has_bias):
    tmp_layer = input_layer
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample, activation=None, border_mode=border_mode, name=layer_name, bias=bias, init=init_, W_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = Scaling(name=layer_name + '_scale')(tmp_layer)
    tmp_layer = Lambda(lambda x: tf.nn.elu(x), name=layer_name + '_nonlin')(tmp_layer)
    return tmp_layer

def add_inception(input_layer, list_nb_filter, base_name):
    tower_1_1 = bnConv_layer(input_layer=input_layer, layer_name=base_name + '_1x1', nb_filter=list_nb_filter[0], nb_row=1, nb_col=1)

    tower_2_1 = bnConv_layer(input_layer=input_layer, layer_name=base_name + '_3x3_reduce', nb_filter=list_nb_filter[1], nb_row=1, nb_col=1)
    tower_2_2 = bnConv_layer(input_layer=tower_2_1, layer_name=base_name + '_3x3', nb_filter=list_nb_filter[2], nb_row=3, nb_col=3)

    tower_3_1 = bnConv_layer(input_layer=input_layer, layer_name=base_name + '_5x5_reduce',nb_filter=list_nb_filter[3],nb_row=1, nb_col=1)
    tower_3_2 = bnConv_layer(input_layer=tower_3_1, layer_name=base_name + '_5x5', nb_filter=list_nb_filter[4], nb_row=5, nb_col=5)

    tower_4_1 = MaxPooling2D(name=base_name + '_pool',pool_size=(3, 3), strides=(1, 1), border_mode='same')(input_layer)
    tower_4_2 = bnConv_layer(input_layer=tower_4_1, layer_name=base_name + '_pool_proj', nb_filter=list_nb_filter[5],nb_row=1, nb_col=1)

    return merge(inputs=[tower_1_1,tower_2_2,tower_3_2,tower_4_2], mode='concat',name=base_name + '_output')

class Scaling(Layer):
    def __init__(self, init_weights=1.0, bias=True, trainable=True, **kwargs):
        self.supports_masking = True
        self.initial_weights = init_weights
        self.has_bias = bias
        self.trainable = trainable
        super(Scaling, self).__init__(**kwargs)

    def build(self, input_shape):
        size = input_shape[-1]  # Tensorflow
        self.scaling_factor = self.add_weight(shape=(1, 1, size), initializer='one', trainable=self.trainable, name=None)

        if (self.has_bias):
            self.bias = self.add_weight(shape=(1, 1, size), initializer='zero', trainable=self.trainable, name= None )
            self.trainable_weights = [self.scaling_factor, self.bias]
        else:
            self.trainable_weights = [self.scaling_factor]

    def call(self, x, mask=None):
        out = self.scaling_factor * x
        if (self.has_bias):
            out = out + self.bias
        return out

    def get_config(self):
        config = {}
        config['scaling_factor'] = self.scaling_factor
        if (self.has_bias):
            config['bias'] = self.bias
        base_config = super(Scaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
