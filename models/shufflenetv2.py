import numpy as np
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    Concatenate,
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Lambda,
    DepthwiseConv2D,
    UpSampling2D,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Adam

from .metrics import mean_iou


def channel_shuffle(x):
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_split = in_channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, in_channels])
    
    return x


def channel_split(x, name=''):
    in_channels = x.shape.as_list()[-1]
    ip = in_channels // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='{}/sp{}_slice'.format(name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='{}/sp{}_slice'.format(name, 1))(x)
    return c_hat, c


def shuffle_unit(inp, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    prefix = 'stage{}/block{}'.format(stage, block)
    bn_axis = -1

    bottleneck_channels = int(out_channels * bottleneck_ratio) // 2
    if strides < 2:
        c_hat, c = channel_split(inp, '{}/spl'.format(prefix))
        inp = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inp)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    # Special modification of Depth2D from paper
    x = DepthwiseConv2D(kernel_size=3, strides=(1 if stage == 4 else strides),
                        dilation_rate=(2 if stage == 4 and block > 1 else 1),
                        padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        # Special modification of Depth2D from paper
        s2 = DepthwiseConv2D(kernel_size=3, strides=(1 if stage == 4 else strides),
                             padding='same', name='{}/3x3dwconv_2'.format(prefix))(inp)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def stage(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                     strides=2, bottleneck_ratio=bottleneck_ratio,
                     stage=stage, block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=1,
                         bottleneck_ratio=bottleneck_ratio,
                         stage=stage, block=1+i)

    return x


# shufflenet V2 encoder
def _shufflenetv2(inp):
    shuffle_units = [3, 7, 3]
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
    scale_factor = 1.0

    # bottleneck_ratio = alpha
    bottleneck_ratio = 0.5

    exp = np.insert(np.arange(len(shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]
    out_channels_in_stage[0] = 24  # first unit has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    x = Conv2D(out_channels_in_stage[0], 3, padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(inp)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    for i in range(len(shuffle_units)):
        repeat = shuffle_units[i]
        x = stage(x, out_channels_in_stage, repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio, stage=i+2)

    # x = Conv2D(1, (1, 1), padding='same', name='conv_1c_1x1')(x)
    return x


def deeplabv3plus(x):
    bn_axis = -1
    # dense prediction cell
    start = Conv2D(128, 3, dilation_rate=(1, 6), padding='same', name='dpc_start')(x)
    start = BatchNormalization(axis=bn_axis, name='dpc_batch_start')(start)
    start = Activation('relu', name='dpc_act_start')(start)
   
    a = Conv2D(64, 3, dilation_rate=(18, 15), padding='same', name='dpc_a1')(start)
    a = BatchNormalization(axis=bn_axis, name='dpc_batch_a1')(a)
    a = Activation('relu', name='dpc_act_a1')(a)

    a2 = Conv2D(64, 3, dilation_rate=(6, 3), padding='same', name='dpc_a2')(a)
    a2 = BatchNormalization(axis=bn_axis, name='dpc_batch_a2')(a2)
    a2 = Activation('relu', name='dpc_act_a2')(a2)

    b = Conv2D(128, 3, dilation_rate=(6, 21), padding='same', name='dpc_b1')(start)
    b = BatchNormalization(axis=bn_axis, name='dpc_batch_b1')(b)
    b = Activation('relu', name='dpc_act_b1')(b)

    c = Conv2D(128, 3, dilation_rate=(1, 1), padding='same', activation='relu', name='dpc_c1')(start)
    c = BatchNormalization(axis=bn_axis, name='dpc_batch_c1')(c)
    c = Activation('relu', name='dpc_act_c1')(c)

    x = Concatenate(name='dpc_concat')([a, a2, b, start, c])

    # convs
    x = Conv2D(256, kernel_size=1, padding='same', use_bias=False, activation='relu', name='deeplab_conv1')(x)
    x = Conv2D(1, kernel_size=1, padding='same', use_bias=False, activation='relu', name='deeplab_conv2')(x)

    return x


def shufflenetv2(input_size=(480, 640, 3), loss='binary_crossentropy'):
    inp = Input(input_size)

    x = _shufflenetv2(inp)
    o = deeplabv3plus(x)

    # note: not using dropout here
    o = UpSampling2D(size=(16, 16), interpolation='bilinear')(o)
    o = Conv2D(1, (3, 3), padding='same', name='conv_final', activation='sigmoid')(o) 

    model = Model(inp, o, name='shufflenetv2')
    model.compile(optimizer=Adam(lr=0.001, decay=0.0005), loss=loss,
                  metrics=['accuracy', mean_iou])

    return model 
