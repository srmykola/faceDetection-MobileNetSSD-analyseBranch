from keras.layers import Input

from nnBlocks import separable_res_block1, relu6, DepthwiseConv2D
from nnBlocks import _conv_block, bn_conv, bn_conv_layer
from nnBlocks import add_inception, Scaling
from depthwiseBlocks import depthwiseConvBlockClassification

def mobileNetV1( inputShape, alpha = 1.0, depthMultiplier = 1, imgHeight = 512, imgWidth = 512, imgChannels = 3 ):

    imgInput = Input( shape = inputShape )

    x = Lambda( lambda z: z/255., output_shape = ( imgHeight, imgWidth, imgChannels ),
               name = 'lambda1' )( imgInput ) # Convert input feature range to [-1,1]

    x = Lambda( lambda z: z - 0.5, output_shape = ( imgHeight, imgWidth, imgChannels ),
                   name = 'lambda2' )( x ) # Convert input feature range to [-1,1]

    x = Lambda( lambda z: z * 2., output_shape = ( imgHeight, imgWidth, imgChannels ),
                   name = 'lambda3' )( x ) # Convert input feature range to [-1,1]

    x = _conv_block(x, 32, alpha, strides=(2, 2))
    x = depthwiseConvBlockClassification(x, 64, alpha, depthMultiplier, block_id=1)
    x = depthwiseConvBlockClassification(x, 128, alpha, depthMultiplier, strides=(2, 2), block_id=2)
    x = depthwiseConvBlockClassification(x, 128, alpha, depthMultiplier, block_id=3)
    x = depthwiseConvBlockClassification(x, 256, alpha, depthMultiplier, strides=(2, 2), block_id=4)
    x = depthwiseConvBlockClassification(x, 256, alpha, depthMultiplier, block_id=5)
    x = depthwiseConvBlockClassification(x, 512, alpha, depthMultiplier, strides=(2, 2), block_id=6)
    x = depthwiseConvBlockClassification(x, 512, alpha, depthMultiplier, block_id=7)
    x = depthwiseConvBlockClassification(x, 512, alpha, depthMultiplier, block_id=8)
    x = depthwiseConvBlockClassification(x, 512, alpha, depthMultiplier, block_id=9)
    analyseBranchIntersection = depthwiseConvBlockClassification(x, 512, alpha, depthMultiplier, block_id=10)
    conv4_3 = depthwiseConvBlockClassification( analyseBranchIntersection, 512, alpha, depthMultiplier, block_id = 11 ) #11 conv4_3 (300x300)-> 19x19

    x = depthwiseConvBlockClassification( conv4_3, 1024, alpha, depthMultiplier, strides = (2, 2), block_id = 12 )   # (300x300) -> 10x10
    fc7 = depthwiseConvBlockClassification( x, 1024, alpha, depthMultiplier, block_id = 13 ) # 13 fc7 (300x300) -> 10x10

    return analyseBranchIntersection, conv4_3, fc7, imgInput
