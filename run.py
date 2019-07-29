import datetime
import glob

import click
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint

from data_generator import *
import models


AVAILABLE_MODELS = ['unet', 'fcn_vgg16_32s']
MODEL_MAPPING = {
    'unet': models.unet,
    'fcn_vgg16_32s': models.fcn_vgg16_32s
}


IMG_TARGET_SIZE = (240, 320)
RESIZE_TO = tuple(reversed(IMG_TARGET_SIZE))
INPUT_SIZE = (240, 320, 3)
BATCH_SIZE = 15
N_TRAIN_SAMPLES = len(glob.glob('data/train/image/*.png', recursive=False))
N_VAL_SAMPLES = len(glob.glob('data/val/image/*.png', recursive=False))
N_TEST_SAMPLES = len(glob.glob('data/test/image/*.png', recursive=False))


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
def train(model, gen, plot, aug, epochs):
    date = datetime.datetime.now()
    now_str = date.strftime('%Y-%m-%d-%H%M%S')

    model_filename = '{}_{}_{}'.format(model, str(gen), now_str)

    model_out = 'trained_models/{}.hdf5'.format(model_filename)
    model_checkpoint = ModelCheckpoint(model_out, monitor='loss',
                                       verbose=1, save_best_only=True)

    model = MODEL_MAPPING[model](input_size=INPUT_SIZE)

    if gen:
        #data_gen_args = dict(zoom_range=0.05, horizontal_flip=True)
        # TODO: augmentation
        my_data_gen = train_generator(BATCH_SIZE, 'data/train', 'image',
                                      'masks',
                                      img_target_size=IMG_TARGET_SIZE, augs={})
        val_data_gen = train_generator(1, 'data/val', 'image', 'masks',
                                       img_target_size=IMG_TARGET_SIZE)

        steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE

        history = model.fit_generator(my_data_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      callbacks=[model_checkpoint],
                                      validation_data=val_data_gen,
                                      validation_steps=N_VAL_SAMPLES)
    else:
        X_train, Y_train = load_data_memory(['data/train', 'data/val'],
                                            'image', 'masks',
                                            resize=RESIZE_TO)
        history = model.fit(X_train, Y_train,
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
        plt.legend(loc='best')
        plt.savefig('plots/acc_{}.png'.format(model_filename))


def evaluate():
    pass
    #model = unet(pretrained_weights='unet_zajac.hdf5', input_size=INPUT_SIZE)
    #model = unet(input_size=INPUT_SIZE)
    #eval_gen = eval_generator(1, 'data/test', 'image', 'masks', img_target_size=IMG_TARGET_SIZE)
    #loss, acc = model.evaluate_generator(eval_gen, steps=N_TEST_SAMPLES, verbose=0)
    #print('Eval loss, acc: ', loss, acc)


def predict():
    pass
    #test_gen = test_data_generator('data/test', 'image', img_target_size=IMG_TARGET_SIZE)
    #results = model.predict_generator(test_gen, steps=N_TEST_SAMPLES, verbose=1)
    #save_predicted_images('data/predictions', 'data/test/image', results, IMG_TARGET_SIZE)


cli.add_command(train)


if __name__ == '__main__':
    cli()
