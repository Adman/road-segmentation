import datetime
import glob

import click
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import keras.backend as K
from keras.callbacks import ModelCheckpoint

import segmentation_models as sm

from data_generator import (
    train_generator,
    test_data_generator,
    eval_generator,
    save_predicted_images,
    load_data_memory,
)
import models


AVAILABLE_MODELS = ['unet', 'fcn_vgg16_32s', 'segnet', 'resnet',
                    'unet_resnet34', 'unet_resnet50', 'unet_vgg16',
                    'linknet_vgg16']
MODEL_MAPPING = {
    'unet': models.unet,
    'fcn_vgg16_32s': models.fcn_vgg16_32s,
    'segnet': models.segnet,
    'resnet': models.resnet,
 
    'unet_resnet34': models.unet_resnet34,
    'unet_resnet50': models.unet_resnet50,
    'unet_vgg16': models.unet_vgg16,

    'linknet_vgg16': models.linknet_vgg16
}


IMG_TARGET_SIZE = (480, 640)
RESIZE_TO = tuple(reversed(IMG_TARGET_SIZE))
INPUT_SIZE = IMG_TARGET_SIZE + (3,)
BATCH_SIZE = 2
N_TRAIN_SAMPLES = len(glob.glob('data/train/image/*.png', recursive=False))
N_VAL_SAMPLES = len(glob.glob('data/val/image/*.png', recursive=False))
N_TEST_SAMPLES = len(glob.glob('data/test/image/*.png', recursive=False))
LOSS = models.metrics.dice_coef_loss #'binary_crossentropy'


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
@click.option('--aug', '-a', type=bool, default=False,
              help='Whether to use data augmentation')
@click.option('--epochs', '-e', type=int, default=3,
              help='Number of epochs')
@click.option('--hsv', '-h', type=bool, default=False,
              help='Whether to convert rgb image to hsv')
def train(model, gen, plot, aug, epochs, hsv):
    date = datetime.datetime.now()
    now_str = date.strftime('%Y-%m-%d-%H%M%S')

    model_filename = '{}_{}_{}_{}x{}_{}'.format(model,
                                          'gen' if gen else 'nogen',
                                          'hsv' if hsv else 'rgb',
                                          IMG_TARGET_SIZE[1],
                                          IMG_TARGET_SIZE[0],
                                          now_str)

    model_out = 'trained_models/{}.hdf5'.format(model_filename)
    model_checkpoint = ModelCheckpoint(model_out, monitor='loss',
                                       verbose=1, save_best_only=True)

    _model = MODEL_MAPPING[model](input_size=INPUT_SIZE,
                                  loss=LOSS)

    if gen:
        data_gen_args = {}
        if aug:
            data_gen_args = dict(fill_mode='constant',
                                 #zoom_range=0.05,
                                 rotation_range=20,
                                 #vertical_flip=True,
                                 horizontal_flip=True)

        my_data_gen = train_generator(BATCH_SIZE, 'data/train', 'image',
                                      'masks',
                                      img_target_size=IMG_TARGET_SIZE,
                                      augs=data_gen_args,
                                      tohsv=hsv,
                                      aug=aug)

        X_val, Y_val = load_data_memory(['data/val'], 'image', 'masks',
                                        resize=RESIZE_TO, tohsv=hsv)

        steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE
        if aug and not hsv:
            steps_per_epoch *= 3

        history = _model.fit_generator(my_data_gen,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       callbacks=[model_checkpoint],
                                       validation_data=(X_val, Y_val))
    else:
        X_train, Y_train = load_data_memory(['data/train', 'data/val'],
                                            'image', 'masks',
                                            resize=RESIZE_TO,
                                            tohsv=hsv)

        history = _model.fit(X_train, Y_train,
                             epochs=epochs,
                             batch_size=BATCH_SIZE,
                             callbacks=[model_checkpoint],
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
    _model = _model(pretrained_weights=path, input_size=INPUT_SIZE,
                    loss=LOSS)
    eval_gen = eval_generator(1, 'data/train', 'image', 'masks',
                              img_target_size=IMG_TARGET_SIZE,
                              tohsv=hsv)

    loss, acc, miou, _ = _model.evaluate_generator(eval_gen, steps=N_TRAIN_SAMPLES,
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
    _model = _model(pretrained_weights=path, input_size=INPUT_SIZE,
                    loss=LOSS)
    test_gen = test_data_generator('data/test', 'image',
                                   img_target_size=IMG_TARGET_SIZE,
                                   tohsv=hsv)
    results = _model.predict_generator(test_gen, steps=N_TEST_SAMPLES,
                                       verbose=1)

    save_predicted_images('data/predictions', 'data/test/image', results,
                          IMG_TARGET_SIZE)


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(predict)


if __name__ == '__main__':
    cli()
