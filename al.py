import datetime
import click
import os
import glob
import shutil

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from alutils import (
    strategy_random,

    stopping_nostop,    
)

from run import (
    MODEL_MAPPING,
    AVAILABLE_MODELS,
    TRAIN_DIR,
    VAL_DIR,
    INPUT_SIZE,
    IMG_TARGET_SIZE,
    RESIZE_TO,
    BATCH_SIZE,
    LOSS,
)
from data_generator import (
    train_generator,
    load_data_memory,
)


ACTIVE_DIR = 'data/train_active'
ACTIVE_IMAGE_DIR = os.path.join(ACTIVE_DIR, 'image')
ACTIVE_MASK_DIR = os.path.join(ACTIVE_DIR, 'masks')

INIT_PICK = 30
TRAIN_PICK = 10
EPOCHS_PER_ROUND = 3


def pick_copy_images(images, k=1, strategy=strategy_random, **kwargs):
    """Pick k random images and copy them to ACTIVE_FOLDER.
       Picked images are removed from `images` list.
    """
    k = min(k, len(images))
    imgs = strategy(images, k, **kwargs)

    for img in imgs:
        iname = os.path.basename(img)
        mask = os.path.join(TRAIN_DIR, 'masks', iname)

        shutil.copyfile(img, os.path.join(ACTIVE_IMAGE_DIR, iname))
        shutil.copyfile(mask, os.path.join(ACTIVE_MASK_DIR, iname))
        images.remove(img)

    return imgs


@click.group()
def cli():
    pass


STRATEGY_MAPPING = {
    'random': strategy_random
}
STOPPINGS_MAPPING = {
    'nostop': stopping_nostop
}
AVAILABLE_STRATEGIES = list(STRATEGY_MAPPING.keys())
AVAILABLE_STOPPINGS = list(STOPPINGS_MAPPING.keys())


@click.command(help='Run active learning simulation')
@click.option('--model', '-m', type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to simulate training on')
@click.option('--pick', '-p', type=click.Choice(AVAILABLE_STRATEGIES),
              default='random', help='Strategy to pick images into training')
@click.option('--stopping', '-s', type=click.Choice(AVAILABLE_STOPPINGS),
              default='nostop', help='Method which stops training')
def simulate(model, pick, stopping):
    date = datetime.datetime.now()
    now_str = date.strftime('%Y-%m-%d-%H%M%S')

    # TODO: put all hyperparams to filename
    model_filename = '{}_i{}xp{}xe{}_{}_{}_{}'.format(model,
                                                      INIT_PICK,
                                                      TRAIN_PICK,
                                                      EPOCHS_PER_ROUND,
                                                      pick,
                                                      stopping,
                                                      now_str)

    model_out = 'trained_active_models/{}.hdf5'.format(model_filename)
    # TODO: test EarlyStopping
    tensorboard = TensorBoard(log_dir='./logs/{0}'.format(model_filename),
                              histogram_freq=0,
                              write_graph=True, write_images=True,
                              update_freq='epoch')

    _model = MODEL_MAPPING[model](input_size=INPUT_SIZE, loss=LOSS)

    pick_strategy = STRATEGY_MAPPING[pick]
    stop_strategy = STOPPINGS_MAPPING[stopping]

    data_gen_args = dict(fill_mode='constant',
                         #zoom_range=0.05,
                         rotation_range=5,
                         #vertical_flip=True,
                         horizontal_flip=True)

    X_val, Y_val = load_data_memory([VAL_DIR], 'image', 'masks',
                                    resize=RESIZE_TO, aug=True)

    if not os.path.exists(os.path.join(ACTIVE_DIR, 'image')):
        os.makedirs(os.path.join(ACTIVE_DIR, 'image'))
    if not os.path.exists(os.path.join(ACTIVE_DIR, 'masks')):
        os.makedirs(os.path.join(ACTIVE_DIR, 'masks'))

    unpicked = glob.glob(os.path.join(TRAIN_DIR, 'image/*.png'))
    _ = pick_copy_images(unpicked, INIT_PICK, strategy=pick_strategy)
    n_train_samples = INIT_PICK

    my_data_gen = train_generator(BATCH_SIZE, ACTIVE_DIR,
                                  'image', 'masks',
                                  img_target_size=IMG_TARGET_SIZE,
                                  augs=data_gen_args)

    histories = []
    while True:
        steps_per_epoch = (n_train_samples // BATCH_SIZE) * 3

        # refresh generator to incorporate newly picked images
        my_data_gen = train_generator(BATCH_SIZE, ACTIVE_DIR,
                                      'image', 'masks',
                                      img_target_size=IMG_TARGET_SIZE,
                                      augs=data_gen_args)

        print('Starting new round with {} images'.format(n_train_samples))
        history = _model.fit_generator(my_data_gen,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=EPOCHS_PER_ROUND,
                                       callbacks=[tensorboard],
                                       validation_data=(X_val, Y_val))           
        
        histories.append(history)

        # pick next
        if not unpicked or stop_strategy(history=histories):
            break

        _ = pick_copy_images(unpicked, TRAIN_PICK)
        n_train_samples += TRAIN_PICK



cli.add_command(simulate)


if __name__ == '__main__':
    cli()
