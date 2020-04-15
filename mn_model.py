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
from nnBlocks import separable_res_block1, relu6, DepthwiseConv2D
from nnBlocks import _conv_block, bn_conv, bn_conv_layer
from nnBlocks import add_inception, Scaling
from depthwiseBlocks import depthwiseConvBlockDetection, depthwiseConvBlockClassification

mobilenet = True
separable_filter = False
conv_model = False
# tf = K.tf
dropout_rate = 0.55
W_regularizer = None
init_ = 'glorot_uniform'
conv_has_bias = True #False for BN
fc_has_bias = True




def mn_model(image_size,
                n_classes,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                limit_boxes=True,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False):

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios_conv4_3 = aspect_ratios_per_layer[0]
        aspect_ratios_fc7 = aspect_ratios_per_layer[1]
        aspect_ratios_conv6_2 = aspect_ratios_per_layer[2]
        aspect_ratios_conv7_2 = aspect_ratios_per_layer[3]
        aspect_ratios_conv8_2 = aspect_ratios_per_layer[4]
        aspect_ratios_conv9_2 = aspect_ratios_per_layer[5]
    else:
        aspect_ratios_conv4_3 = aspect_ratios_global
        aspect_ratios_fc7 = aspect_ratios_global
        aspect_ratios_conv6_2 = aspect_ratios_global
        aspect_ratios_conv7_2 = aspect_ratios_global
        aspect_ratios_conv8_2 = aspect_ratios_global
        aspect_ratios_conv9_2 = aspect_ratios_global

     # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv4_3 = n_boxes[0] # 4 boxes per cell for the original implementation
        n_boxes_fc7 = n_boxes[1] # 6 boxes per cell for the original implementation
        n_boxes_conv6_2 = n_boxes[2] # 6 boxes per cell for the original implementation
        n_boxes_conv7_2 = n_boxes[3] # 6 boxes per cell for the original implementation
        n_boxes_conv8_2 = n_boxes[4] # 4 boxes per cell for the original implementation
        n_boxes_conv9_2 = n_boxes[5] # 4 boxes per cell for the original implementation
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes_conv4_3 = n_boxes
        n_boxes_fc7 = n_boxes
        n_boxes_conv6_2 = n_boxes
        n_boxes_conv7_2 = n_boxes
        n_boxes_conv8_2 = n_boxes
        n_boxes_conv9_2 = n_boxes


    print ("Height, Width, Channels :", image_size[0], image_size[1], image_size[2])
       # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    input_shape = (img_height, img_width, img_channels)

    img_input = Input(shape=input_shape)

    alpha = 1.0
    depth_multiplier = 1


    x = Lambda(lambda z: z/255., # Convert input feature range to [-1,1]
              output_shape=(img_height, img_width, img_channels),
               name='lambda1')(img_input)
    x = Lambda(lambda z: z - 0.5, # Convert input feature range to [-1,1]
                  output_shape=(img_height, img_width, img_channels),
                   name='lambda2')(x)
    x = Lambda(lambda z: z *2., # Convert input feature range to [-1,1]
                  output_shape=(img_height, img_width, img_channels),
                   name='lambda3')(x)

    x = _conv_block(x, 32, alpha, strides=(2, 2))
    x = depthwiseConvBlockClassification(x, 64, alpha, depth_multiplier, block_id=1)

    x = depthwiseConvBlockClassification(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = depthwiseConvBlockClassification(x, 128, alpha, depth_multiplier, block_id=3)

    x = depthwiseConvBlockClassification(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = depthwiseConvBlockClassification(x, 256, alpha, depth_multiplier, block_id=5)

    x = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier, block_id=10)
    conv4_3 = depthwiseConvBlockClassification(x, 512, alpha, depth_multiplier, block_id=11) #11 conv4_3 (300x300)-> 19x19

    x = depthwiseConvBlockClassification(conv4_3, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)   # (300x300) -> 10x10
    fc7 = depthwiseConvBlockClassification(x, 1024, alpha, depth_multiplier, block_id=13) # 13 fc7 (300x300) -> 10x10


    conv6_1 = bn_conv(fc7, 'detection_conv6_1', 256, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    conv6_2 = depthwiseConvBlockDetection(input = conv6_1, layer_name='detection_conv6_2', strides=(2,2),
                                        pointwise_conv_filters=512, alpha=alpha, depth_multiplier=depth_multiplier,
                                        padding = 'same', use_bias = True, block_id=1)

    conv7_1 = bn_conv(conv6_2, 'detection_conv7_1', 128, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    conv7_2 = depthwiseConvBlockDetection(input = conv7_1, layer_name='detection_conv7_2', strides=(2,2),
                                        pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier,
                                        padding = 'same', use_bias = True, block_id=2)
    #conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='detection_conv7_1')(conv6_2)
    #conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='detection_conv7_2')(conv7_1)

    conv8_1 = bn_conv(conv7_2, 'detection_conv8_1', 128, 1, 1, subsample =(1,1), border_mode ='same', bias=conv_has_bias)

    conv8_2 = depthwiseConvBlockDetection(input = conv8_1, layer_name='detection_conv8_2', strides=(2,2),
                            pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier,
                            padding = 'same', use_bias = True, block_id=3)

    # # conv8_2 = bn_conv(conv8_1, 'detection_conv8_2', 256, 2, 2, subsample =(1,1), border_mode ='same', bias=conv_has_bias)

    conv9_1 = bn_conv(conv8_2, 'detection_conv9_1', 64, 1, 1,  subsample =(1,1), border_mode ='same', bias=conv_has_bias)
    # conv9_2 = bn_conv(conv9_1, 'detection_conv9_2', 128, 3, 3, subsample =(2,2), border_mode ='same', bias=conv_has_bias)

    conv9_2 = depthwiseConvBlockDetection(input = conv9_1, layer_name='detection_conv9_2', strides=(2,2),
                                    pointwise_conv_filters=256, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=4)


    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='detection_conv4_3_norm')(conv4_3)


    conv4_3_norm_mbox_conf = depthwiseConvBlockDetection(input = conv4_3_norm, layer_name='detection_conv4_3_norm_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv4_3 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=1)


    fc7_mbox_conf = depthwiseConvBlockDetection(input = fc7, layer_name='detection_fc7_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_fc7 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=2)
    conv6_2_mbox_conf = depthwiseConvBlockDetection(input = conv6_2, layer_name='detection_conv6_2_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv6_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=3)

    conv7_2_mbox_conf = depthwiseConvBlockDetection(input = conv7_2, layer_name='detection_conv7_2_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv7_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=4)

    conv8_2_mbox_conf = depthwiseConvBlockDetection(input = conv8_2, layer_name='detection_conv8_2_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv8_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=5)
    conv9_2_mbox_conf = depthwiseConvBlockDetection(input = conv9_2, layer_name='detection_conv9_2_mbox_conf', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv9_2 * n_classes, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=6)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`

    conv4_3_norm_mbox_loc = depthwiseConvBlockDetection(input = conv4_3_norm, layer_name='detection_conv4_3_norm_mbox_loc', strides=(1,1),
                                    pointwise_conv_filters=n_boxes_conv4_3 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                    padding = 'same', use_bias = True, block_id=1)

    fc7_mbox_loc = depthwiseConvBlockDetection(input = fc7, layer_name='detection_fc7_mbox_loc', strides=(1,1),
                                pointwise_conv_filters=n_boxes_fc7 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                padding = 'same', use_bias = True, block_id=2)


    conv6_2_mbox_loc = depthwiseConvBlockDetection(input = conv6_2, layer_name='detection_conv6_2_mbox_loc', strides=(1,1),
                                pointwise_conv_filters=n_boxes_conv6_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                padding = 'same', use_bias = True, block_id=3)

    conv7_2_mbox_loc = depthwiseConvBlockDetection(input = conv7_2, layer_name='detection_conv7_2_mbox_loc', strides=(1,1),
                                pointwise_conv_filters=n_boxes_conv7_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                padding = 'same', use_bias = True, block_id=4)

    conv8_2_mbox_loc = depthwiseConvBlockDetection(input = conv8_2, layer_name='detection_conv8_2_mbox_loc', strides=(1,1),
                                pointwise_conv_filters=n_boxes_conv8_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                padding = 'same', use_bias = True, block_id=5)

    conv9_2_mbox_loc = depthwiseConvBlockDetection(input = conv9_2, layer_name='detection_conv9_2_mbox_loc', strides=(1,1),
                                pointwise_conv_filters=n_boxes_conv9_2 * 4, alpha=alpha, depth_multiplier=depth_multiplier,
                                padding = 'same', use_bias = True, block_id=5)
    ### Generate the anchor boxes

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`


    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios_conv4_3,
                                             two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios_fc7,
                                    two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_fc7_mbox_priorbox')(fc7)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios_conv6_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv6_2_mbox_priorbox')(conv6_2)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios_conv7_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv7_2_mbox_priorbox')(conv7_2)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios_conv8_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv8_2_mbox_priorbox')(conv8_2)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios_conv9_2,
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='detection_conv9_2_mbox_priorbox')(conv9_2)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='detection_fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='detection_mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='detection_mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='detection_mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='detection_mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='detection_predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=img_input, outputs=predictions)
    #model = Model(inputs=img_input, outputs=predictions)



    # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
    # be able to generate the default boxes for the matching process outside of the model during training.
    # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
    # Instead, we'll do it in the batch generator function.
    # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.

    predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                 fc7_mbox_conf._keras_shape[1:3],
                                 conv6_2_mbox_conf._keras_shape[1:3],
                                 conv7_2_mbox_conf._keras_shape[1:3],
                                 conv8_2_mbox_conf._keras_shape[1:3],
                                 conv9_2_mbox_conf._keras_shape[1:3]])

    model_layer = dict([(layer.name, layer) for layer in model.layers])

    # for key in model_layer:
    #    model_layer[key].trainable = True


    # model = Model(img_input, conv9_2)
    # model_layer = dict([(layer.name, layer) for layer in model.layers])
    # predictor_sizes = 0

    return model, model_layer, img_input, predictor_sizes
