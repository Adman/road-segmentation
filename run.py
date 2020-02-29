import datetime
import glob
import os

import click
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np

import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Model

import segmentation_models as sm
import matplotlib.pyplot as plt

from data_generator import (
    train_generator,
    test_data_generator,
    eval_generator,
    save_predicted_images,
    load_data_memory,
    load_single_image,
)
import models


MODEL_MAPPING = {
    'unet': models.unet,
    'fcn_vgg16_32s': models.fcn_vgg16_32s,
    'segnet': models.segnet,
    'resnet': models.resnet,
    'segnetsmall': models.segnetsmall,
    'resnetsmall': models.resnetsmall,
    'resnet_bnn': models.resnet_bnn,

    'shuffleseg': models.shuffleseg,
    'shufflenetv2': models.shufflenetv2,
    'mobilenetv2': models.mobilenetv2,

    'segnet_mobilenet': models.segnet_mobilenet,

    'unet_resnet34': models.unet_resnet34,
    'unet_resnet50': models.unet_resnet50,
    'unet_vgg16': models.unet_vgg16,
    'unet_mobilenetv2': models.unet_mobilenetv2,

    'linknet_vgg16': models.linknet_vgg16,
    'linknet_resnet50': models.linknet_resnet50,
    'linknet_mobilenetv2': models.linknet_mobilenetv2,
    'linknet_efficientnetb0': models.linknet_efficientnetb0,
    'linknet_efficientnetb7': models.linknet_efficientnetb7,

    'fpn_resnet34': models.fpn_resnet34,
    'fpn_resnet50': models.fpn_resnet50,
    'fpn_vgg16': models.fpn_vgg16
}

AVAILABLE_MODELS = list(MODEL_MAPPING.keys())

IMG_TARGET_SIZE = (480, 640)
RESIZE_TO = tuple(reversed(IMG_TARGET_SIZE))
INPUT_SIZE = IMG_TARGET_SIZE + (3,)
BATCH_SIZE = 2
N_TRAIN_SAMPLES = len(glob.glob('data/train/image/*.png', recursive=False))
N_VAL_SAMPLES = len(glob.glob('data/val/image/*.png', recursive=False))
N_TEST_SAMPLES = len(glob.glob('data/test/image/*.png', recursive=False))
#LOSS = models.metrics.dice_coef_loss
LOSS = 'binary_crossentropy'

PRODUCTION_DATASET = 'data/datasets/deggendorf'
TRAIN_DIRECTORY = 'data/train'


@click.group()
def cli():
    pass


@click.command(help='Train specified model')
@click.option('--model', '-m', type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to train')
@click.option('--gen', '-g', type=bool, default=True,
              help='Whether to use generator for feeding data')
@click.option('--plot', '-p', type=bool, default=True,
              help='Wheter to plot losses and accuracies')
@click.option('--aug', '-a', type=bool, default=True,
              help='Whether to use data augmentation')
@click.option('--epochs', '-e', type=int, default=3,
              help='Number of epochs')
@click.option('--hsv', '-h', type=bool, default=False,
              help='Whether to convert rgb image to hsv')
@click.option('--weights', '-w', type=str, default='',
              help='Path pretrained weights')
@click.option('--production', type=bool, default=False,
              help='Whether to train on entire dataset for production')
def train(model, gen, plot, aug, epochs, hsv, weights, production):
    global N_TRAIN_SAMPLES
    date = datetime.datetime.now()
    now_str = date.strftime('%Y-%m-%d-%H%M%S')

    model_filename = '{}_{}_{}_{}_{}x{}_{}'.format(model,
                                          'gen' if gen else 'nogen',
                                          'aug' if aug else 'noaug',
                                          'hsv' if hsv else 'rgb',
                                          IMG_TARGET_SIZE[1],
                                          IMG_TARGET_SIZE[0],
                                          now_str)

    model_out = 'trained_models/{}.hdf5'.format(model_filename)
    model_checkpoint = ModelCheckpoint(model_out, monitor='val_loss',
                                       verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=int(epochs*0.05))
    tensorboard = TensorBoard(log_dir='./logs/{0}'.format(model_filename),
                              histogram_freq=0,
                              write_graph=True, write_images=True,
                              update_freq='epoch')

    _model = MODEL_MAPPING[model](input_size=INPUT_SIZE,
                                  loss=LOSS)

    _model.summary()

    if weights != '':
        _model.load_weights(weights)

    train_dir = TRAIN_DIRECTORY
    if production:
        train_dir = PRODUCTION_DATASET
        N_TRAIN_SAMPLES = len(glob.glob(os.path.join(train_dir,
                                                     'image/*.png'),
                                        recursive=False))

    if gen:
        data_gen_args = {}
        if aug:
            data_gen_args = dict(fill_mode='constant',
                                 #zoom_range=0.05,
                                 rotation_range=5,
                                 #vertical_flip=True,
                                 horizontal_flip=True)

        my_data_gen = train_generator(BATCH_SIZE, train_dir,
                                      'image',
                                      'masks',
                                      img_target_size=IMG_TARGET_SIZE,
                                      augs=data_gen_args,
                                      tohsv=hsv,
                                      aug=aug)

        X_val, Y_val = load_data_memory(['data/val'], 'image', 'masks',
                                        resize=RESIZE_TO, tohsv=hsv, aug=aug)

        steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE
        if aug:
            steps_per_epoch *= 3

        history = _model.fit_generator(my_data_gen,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       callbacks=[model_checkpoint, early_stopping,
                                                  tensorboard],
                                       validation_data=(X_val, Y_val))
    # mainly just for testing purposes
    else:
        X_train, Y_train = load_data_memory(['data/train', 'data/val'],
                                            'image', 'masks',
                                            resize=RESIZE_TO,
                                            tohsv=hsv)

        history = _model.fit(X_train, Y_train,
                             epochs=epochs,
                             batch_size=BATCH_SIZE,
                             callbacks=[model_checkpoint, early_stopping,
                                        tensorboard],
                             validation_split=0.1)

    if plot:
        plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend(loc='best')
        plt.savefig('plots/loss_{}.png'.format(model_filename))

        plt.figure()
        plt.plot(history.history['acc'], label='train accuracy')
        plt.plot(history.history['val_acc'], label='validation accuracy')
        plt.plot(history.history['mean_iou'], label='train mean IoU')
        plt.plot(history.history['val_mean_iou'], label='validation mean IoU')
        plt.legend(loc='best')
        plt.savefig('plots/acc_{}.png'.format(model_filename))


@click.command(help='Evaluate specified model on a test set')
@click.option('--model', '-m', type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to evaluate')
@click.option('--path', '-p', help='Path to saved model')
@click.option('--hsv', '-h', type=bool, default=False,
              help='Whether to convert rgb image to hsv')
def evaluate(model, path, hsv):
    _model = MODEL_MAPPING[model]
    _model = _model(input_size=INPUT_SIZE, loss=LOSS)

    if (path):
        _model.load_weights(path)

    # _model.summary()

    eval_gen = eval_generator(1, 'data/test', 'image', 'masks',
                              img_target_size=IMG_TARGET_SIZE,
                              tohsv=hsv, aug=True)
    loss, acc, miou = _model.evaluate_generator(eval_gen, steps=3*N_TEST_SAMPLES,
                                                verbose=0)

    print('=======================================')
    print('Evaluation results')
    print('Loss: {}, Acc: {}, Mean IoU: {}'.format(loss, acc, miou))
    print('=======================================')


@click.command(help='Evaluate specified model on a test set')
@click.option('--model', '-m', type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to evaluate')
@click.option('--path', '-p', help='Path to saved model')
@click.option('--hsv', '-h', type=bool, default=False,
              help='Whether to convert rgb image to hsv')
def predict(model, path, hsv):
    _model = MODEL_MAPPING[model]
    _model = _model(input_size=INPUT_SIZE, loss=LOSS)

    if (path):
        _model.load_weights(path)

    test_gen = test_data_generator('data/test', 'image',
                                   img_target_size=IMG_TARGET_SIZE,
                                   tohsv=hsv)
    results = _model.predict_generator(test_gen, steps=N_TEST_SAMPLES,
                                       verbose=1)

    save_predicted_images('data/predictions', 'data/test/image', results,
                          RESIZE_TO)


@click.command(help='Vizualize activations of given model at specific layer')
@click.option('--model', '-m', type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to vizualize layers from')
@click.option('--path', '-p', help='Path to saved model')
@click.option('--hsv', '-h', type=bool, default=False,
              help='Whether to convert rgb image to hsv')
@click.option('--img', '-i', help='Path to image')
@click.option('--layer', '-l', type=int, help='Which layer\' activations to visualize')
def vizualize(model, path, hsv, img, layer):
    _model = MODEL_MAPPING[model]
    _model = _model(input_size=INPUT_SIZE, loss=LOSS)

    if (path):
        _model.load_weights(path)

    output_ids = [layer]
    outputs = [_model.layers[i].output for i in output_ids]

    _model = Model(inputs=_model.inputs, outputs=outputs)

    i = load_single_image(img)
    i = np.expand_dims(i, axis=0)
    feature_maps = _model.predict(i)

    plt.switch_backend('TkAgg')
    height = 4 #8
    width = 8 #16
    for fmap in feature_maps:
        ix = 1
        for _ in range(height):
            for _ in range(width):
                ax = plt.subplot(height, width, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[:, :, ix-1], cmap='gray')
                ix += 1
        plt.show()



cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(predict)
cli.add_command(vizualize)


if __name__ == '__main__':
    cli()
