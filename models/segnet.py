from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *

from .metrics import mean_iou


def segnet(input_size=(480, 640, 3), loss='binary_crossentropy'):
    kernel = 3

    encoding_layers = [
        Conv2D(32, (3, 3), padding='same', input_shape=input_size),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder = Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(size=(2, 2)),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),
        Conv2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(32, (kernel, kernel), padding='same'),
        BatchNormalization(axis=3),
        Activation('relu'),

        Conv2D(1, (1, 1), padding='valid'),
        BatchNormalization(axis=3),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Activation('sigmoid'))
    autoencoder.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False),
                        loss=loss, metrics=['accuracy', mean_iou])

    return autoencoder


# taken from
# https://github.com/BBuf/Keras-Semantic-Segmentation/blob/master/Models/Segnet.py
def segnetsmall(input_size=(480, 640, 3), loss='binary_crossentropy'):
    n_classes = 1
    img_input = Input(input_size)

    # encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = (BatchNormalization())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = (BatchNormalization())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = (BatchNormalization())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = (BatchNormalization())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #decoder
    o = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
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
    model = Model(img_input, o, name='segnetSmall')

    model.compile(optimizer=Adam(lr=0.001, decay=0.0005), loss=loss,
                  metrics=['accuracy', mean_iou])

    return model
