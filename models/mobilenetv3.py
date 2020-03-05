from keras import backend as K
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    Multiply,
    DepthwiseConv2D,
    AveragePooling2D,
    UpSampling2D,
)
from keras.models import Model
from keras.optimizers import Adam

from .shufflenetv2 import deeplabv3plus
from .metrics import mean_iou


# taken from https://github.com/osmr/imgclsmob/blob/master/keras_/kerascv/models/common.py#L19
def round_channels(channels, divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def hswish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def hsigmoid(x):
    return K.relu(x + 3.0, max_value=6.0) / 6.0


def bottleneck(x, out_channels, exp_channels, use_kernel3, strides, activation, use_se_flag, stage, unit):
    prefix = 'stage{}/unit{}'.format(stage, unit)
    bn_axis = -1
    in_channels = x.shape.as_list()[-1]
    residual = (in_channels == out_channels) and (strides == 1)
    inpshape = K.int_shape(x)
    use_exp_conv = exp_channels != out_channels
    mid_channels = exp_channels

    if residual:
        identity = x

    if use_exp_conv:
        x = Conv2D(mid_channels, 1, use_bias=False, name='{}/expconv'.format(prefix))(x)
        x = BatchNormalization(axis=bn_axis, name='{}/expbatch'.format(prefix))(x)
        x = Activation(activation, name='{}/expactiv'.format(prefix))(x)

    if use_kernel3:
        x = DepthwiseConv2D(3, strides=strides, use_bias=False, padding='same', name='{}/dwcov3'.format(prefix))(x)
        x = BatchNormalization(axis=bn_axis, name='{}/dwconv3batch'.format(prefix))(x)
        x = Activation(activation, name='{}/dwconv3activ'.format(prefix))(x)
    else:
        x = DepthwiseConv2D(5, strides=strides, padding='same', use_bias=False, name='{}/dwcov5'.format(prefix))(x)
        x = BatchNormalization(axis=bn_axis, name='{}/dwconv5batch'.format(prefix))(x)
        x = Activation(activation, name='{}/dwconv5activ'.format(prefix))(x)

    if use_se_flag:
        reduction = 4
        c = round_channels(float(mid_channels) / reduction)
        pool_size = x._keras_shape[1:3]
        
        w = AveragePooling2D(pool_size=pool_size, name='{}/seavgpool'.format(prefix))(x)
        w = Conv2D(c, 1, use_bias=True, name='{}/seconv1'.format(prefix))(w)
        w = Activation(activation, name='{}/seactiv'.format(prefix))(w)
        w = Conv2D(mid_channels, 1, use_bias=True, name='{}/seconv2'.format(prefix))(w)
        w = Activation(hsigmoid, name='{}/seactiv2'.format(prefix))(w)       

        x = Multiply(name='{}/semul'.format(prefix))([x, w])

    x = Conv2D(out_channels, 1, use_bias=False, padding='same', name='{}/finalconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/finalbatch'.format(prefix))(x)

    if residual:
        x = Add(name='{}/add'.format(prefix))([x, identity])

    return x


def _mobilenetv3(inp, version, alpha):
    if version == 'small':
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == 'large':
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960

    if alpha != 1.0:
        channels = [[round_channels(cij * alpha) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * alpha) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * alpha)
        if alpha > 1.0:
            final_block_channels = round_channels(final_block_channels * alpha)


    bn_axis = -1
    x = Conv2D(16, 3, strides=2, padding='same', use_bias=False, name='init_conv')(inp)
    x = BatchNormalization(axis=bn_axis, name='init_batchnorm')(x)
    x = Activation(hswish, name='init_activation')(x)

    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            exp_channels_ij = exp_channels[i][j]
            strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
            use_kernel3 = kernels3[i][j] == 1
            activation = "relu" if use_relu[i][j] == 1 else hswish
            use_se_flag = use_se[i][j] == 1
            x = bottleneck(x, out_channels, exp_channels_ij, use_kernel3,
                           strides, activation, use_se_flag, i, j)

    # final block
    x = Conv2D(final_block_channels, 1, padding='same', use_bias=False, name='final_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='final_batchnorm')(x)
    x = Activation(hswish, name='final_activation')(x)

    # final se block
    #reduction = 4
    #c = round_channels(float(final_block_channels) / reduction)
    #pool_size = x._keras_shape[1:3]    

    #w = AveragePooling2D(pool_size=pool_size, name='final_seavgpool')(x)
    #w = Conv2D(c, 1, padding='same', use_bias=True, name='final_seconv1')(w)
    #w = Activation(activation, name='final_seactiv')(w)
    #w = Conv2D(final_block_channels, 1, padding='same', use_bias=True, name='final_seconv2')(w)
    #w = Activation(hsigmoid, name='final_seactiv2')(w)

    #x = Multiply(name='final_semul')([x, w])

    return x


def lite_raspp(model, version):
    bn_axis = -1
    if version == 'small':
        f8_name = 'stage2/unit0/expactiv' #'stage3/unit0/expbatch'
        f16_name = 'stage2/unit4/add' #'final_activation'#'stage3/unit1/add'
        #pool = ()
    elif version == 'large':
        f8_name = 'stage3/unit0/expactiv' #'stage2/unit0/expbatch'
        f16_name = 'stage3/unit5/add' #'final_activation'#'stage2/unit1/add'
        pool = (30, 40)

    out_feature8 = model.get_layer(f8_name).output
    out_feature16 = model.get_layer(f16_name).output

    # branch1
    x1 = Conv2D(128, 1, padding='same', name='lraspp/branch1/conv')(out_feature16)
    x1 = BatchNormalization(axis=bn_axis, name='lraspp/branch1/bn')(x1)
    x1 = Activation('relu', name='lraspp/branch1/activation')(x1)

    # branch2
    x2 = AveragePooling2D(pool_size=pool, name='lraspp/branch2/avgpool')(out_feature16)
    x2 = Conv2D(128, 1, padding='same', name='lraspp/branch2/conv')(x2)
    x2 = Activation('sigmoid', name='lraspp/branch2/activation')(x2)
    x2 = UpSampling2D(size=(30, 40), interpolation='bilinear', name='lraspp/branch2/up')(x2)
    
    # branch3
    x3 = Conv2D(1, 1, padding='same', name='lraspp/branch3/conv')(out_feature8)

    # merge1
    x = Multiply(name='lraspp/mul')([x1, x2])
    x = UpSampling2D(size=(2, 2), name='lraspp/up2')(x)
    x = Conv2D(1, 1, padding='same', name='lraspp/conv')(x)

    # merge2
    x = Add()([x, x3])
    x = UpSampling2D(size=(8, 8), name='lraspp/finalupsample')(x)
    x = Activation('softmax')(x)

    return model.input, x


def mobilenetv3(input_size=(480, 640, 3), loss='binary_crossentropy'):
    version = 'small'
    alpha = 0.75
    inp = Input(input_size)

    x = _mobilenetv3(inp, version, alpha)
    o = deeplabv3plus(x)
    o = UpSampling2D(size=(32, 32), interpolation='bilinear')(o)
    o = Conv2D(1, (3, 3), padding='same', name='conv_output', activation='sigmoid')(o)

    # Lite-RASPP does not work well
    # temporary model
    # model = Model(inp, x, name='mobilenetv3')
    # inp, o = lite_raspp(model, version)
    # o = Conv2D(32, 1, padding='same')(x)
    # o = BatchNormalization(axis=-1)(o)
    # o = Activation('sigmoid')(o)
    # o = UpSampling2D(size=(32, 32), name='lraspp/finalupsample')(o)
    # o = Conv2D(1, 1, padding='same', activation='sigmoid')(o)

    model = Model(inp, o, name='mobilenetv3')
    model.compile(optimizer=Adam(lr=0.001, decay=0.0005), loss=loss,
                  metrics=['accuracy', mean_iou])

    return model
