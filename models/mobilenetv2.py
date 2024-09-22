from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Input,
    ReLU,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .metrics import mean_iou
from .shufflenetv2 import deeplabv3plus


def inverted_res_block(inp, filters=None, alpha=None, stride=None, expansion=None,
                       block_id=None, addition=False):
    prefix = 'block{}'.format(block_id)
    bn_axis = -1
    tchannel = K.int_shape(inp)[bn_axis] * expansion
    cchannel = int(filters * alpha)

    x = Conv2D(tchannel, 1, padding='same', use_bias=False, name='{}|conv1'.format(prefix))(inp)
    x = BatchNormalization(axis=bn_axis, name='{}|bn1'.format(prefix))(x)
    x = ReLU(6., name='{}|relu1'.format(prefix))(x)

    x = DepthwiseConv2D(3, strides=stride, padding='same' if stride == 1 else 'valid',
                        use_bias=False, name='{}|dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}|bn2'.format(prefix))(x)
    x = ReLU(6., name='{}|relu2'.format(prefix))(x)

    x = Conv2D(cchannel, 3, padding='same', use_bias=False, name='{}|conv_out'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis)(x)

    if addition:
        x = Add(name='{}|add'.format(prefix))([inp, x])
           
    return x


def _mobilenetv2(inp):
    alpha = 0.25
    init_block_channels = int(32 * alpha)
    final_block_channels = int(1280 * alpha) if alpha > 1 else 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    channels_per_layer = [16, 24, 32, 64, 96, 160, 320]
    expansions = [1, 6, 6, 6, 6, 6, 6]
    start_stride = [1, 2, 2, 2, 1, 1, 1]
    bn_axis = -1

    # initial block
    x = Conv2D(init_block_channels, 3, strides=2, use_bias=False, padding='same', name='init_conv')(inp)
    x = BatchNormalization(axis=bn_axis, name='init_batchnorm')(x)
    x = ReLU(6., name='init_activation')(x)

    block_id = 0
    # inverted residual blocks
    for i, layer in enumerate(layers):
        for j in range(layer):
            stride = start_stride[i] if j == 0 else 1
            x = inverted_res_block(x, filters=channels_per_layer[i],
                                   alpha=alpha, stride=stride,
                                   expansion=expansions[i], block_id=block_id,
                                   addition=(j > 0))
            block_id += 1

    # expensive block
    #x = Conv2D(final_block_channels, 1, use_bias=False, padding='same', name='out_conv')(x)
    #x = BatchNormalization(axis=bn_axis, name='out_batchnorm')(x)
    #x = ReLU(6., name='out_activation')(x)

    return x


def mobilenetv2(input_size=(480, 640, 3), loss='binary_crossentropy'):
    inp = Input(input_size)

    x = _mobilenetv2(inp)
    o = deeplabv3plus(x)

    o = ZeroPadding2D(padding=((1, 0), (1, 0)))(o)
    o = UpSampling2D(size=(16, 16), interpolation='bilinear')(o)
    o = Conv2D(1, (3, 3), padding='same', name='conv_final', activation='sigmoid')(o)

    model = Model(inp, o, name='mobilenetv2')
    model.compile(
        optimizer=Adam(learning_rate=0.001, weight_decay=0.0005),
        loss=loss,
        metrics=['accuracy', mean_iou]
    )

    return model
