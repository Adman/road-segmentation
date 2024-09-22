from tensorflow.keras import backend as K
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Input,
    UpSampling2D,
    ZeroPadding2D
)
from keras.models import Model
from keras.optimizers import Adam

from .metrics import mean_iou


# taken from
# https://github.com/BBuf/Keras-Semantic-Segmentation/blob/master/Models/MobileNetSegnet.py

def relu6(x):
    return K.relu(x, max_value=6)


# Width Multiplier: Thinner Models
def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=3, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def segnet_mobilenet(input_size=(480, 640, 3), loss='binary_crossentropy'):
    input_width = input_size[1]
    input_height = input_size[0]
    n_classes = 1
    img_input = Input(input_size)
    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3

    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x

    o = f5

    o = Conv2D(512, (3, 3), activation='relu', padding='same')(f4)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(32, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    
    o = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid')(o)
    model = Model(img_input, o, name='segnet_mobilenet')

    model.compile(
        optimizer=Adam(learning_rate=0.001, weight_decay=0.0005),
        loss=loss,
        metrics=['accuracy', mean_iou]
)

    return model
