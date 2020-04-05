import datetime
import click
import os
import glob
import shutil

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from matplotlib import pyplot as plt
plt.switch_backend('agg')

from alutils import (
    strategy_random,
    stopping_nostop,
    epochs_constant
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
    do_evaluate
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
EPOCHS_PER_ROUND_CONSTANT = 3

STRATEGY_MAPPING = {
    'random': strategy_random
}
STOPPINGS_MAPPING = {
    'nostop': stopping_nostop
}
EPOCHS_MAPPING = {
    'constant': epochs_constant
}
AVAILABLE_STRATEGIES = list(STRATEGY_MAPPING.keys())
AVAILABLE_STOPPINGS = list(STOPPINGS_MAPPING.keys())
AVAILABLE_EPOCHS = list(EPOCHS_MAPPING.keys())

NAME_MAPPING = {
    'resnetsmall': 'ResNet-4',
    'mobilenetv3small': 'MNetV3-S-2'
}


def pick_copy_images(images, k=1, strategy=strategy_random, **kwargs):
    """Pick k images and copy them to ACTIVE_FOLDER.
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


def plot_histories(model_list, suffix):
    tloss, vloss, vacc, viou = [], [], [], []
    names = []

    for m in model_list:
        names.append(m.name)
        tloss.append(m.get_history('loss'))
        vloss.append(m.get_history('val_loss'))
        vacc.append(m.get_history('val_acc'))
        viou.append(m.get_history('val_mean_iou'))


    output = '{}_{}'.format('_'.join(names), suffix)

    plt.figure()
    for m, tl, vl in zip(model_list, tloss, vloss):
        plt.plot(tl, label='loss {}'.format(NAME_MAPPING[m.name]))
        plt.plot(vl, label='val loss {}'.format(NAME_MAPPING[m.name]))
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('plots/loss_{}.png'.format(output))
    plt.figure()
    for m, va, vi in zip(model_list, vacc, viou):
        plt.plot(va, label='val acc {}'.format(NAME_MAPPING[m.name]))
        plt.plot(vi, label='val iou {}'.format(NAME_MAPPING[m.name]))
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('plots/acc_{}.png'.format(output))


@click.group()
def cli():
    pass


class AlModel:
    def __init__(self, name, model, model_filename):
        self.name = name
        self.model = model
        self.model_filename = model_filename
        self.finished = False
        self.histories = []

        self.model_out = 'trained_active_models/{}.hdf5'.format(model_filename)
        self.model_checkpoint = ModelCheckpoint(self.model_out, monitor='val_loss',
                                                verbose=1, save_best_only=True)
        self.tensorboard = TensorBoard(log_dir='./logs/{0}'.format(self.model_filename),
                                       histogram_freq=0,
                                       write_graph=True, write_images=True,
                                       update_freq='epoch')

    def fit_generator(self, data_generator, steps_per_epoch, validation_data):
        h = self.model.fit_generator(data_generator,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=1,
                                     callbacks=[self.model_checkpoint, self.tensorboard],
                                     validation_data=validation_data)
        self.histories.append(h)

    def eval(self):
        self.model.load_weights(self.model_out)
        return do_evaluate(self.model, False)

    def get_history(self, param):
        res = []
        for h in self.histories:
            res.extend(h.history[param])
        return res

    def get_best(self, param, greater_better=True):
        fn = max if greater_better else min
        return fn(self.get_history(param))


@click.command(help='Run active learning simulation')
@click.option('--model', '-m', multiple=True,
              type=click.Choice(AVAILABLE_MODELS),
              required=True, help='Model to simulate training on')
@click.option('--pick', '-p', type=click.Choice(AVAILABLE_STRATEGIES),
              default='random', help='Strategy to pick images into training')
@click.option('--stopping', '-s', type=click.Choice(AVAILABLE_STOPPINGS),
              default='nostop', help='Method which stops training')
@click.option('--epochs', '-e', type=click.Choice(AVAILABLE_EPOCHS),
              default='constant', help='Method to choose number epochs for round')
def simulate(model, pick, stopping, epochs):
    data_gen_args = dict(fill_mode='constant',
                         #zoom_range=0.05,
                         rotation_range=5,
                         #vertical_flip=True,
                         horizontal_flip=True)

    pick_strategy = STRATEGY_MAPPING[pick]
    stop_strategy = STOPPINGS_MAPPING[stopping]
    epoch_strategy = EPOCHS_MAPPING[epochs]

    X_val, Y_val = load_data_memory([VAL_DIR], 'image', 'masks',
                                    resize=RESIZE_TO, aug=True)

    if not os.path.exists(os.path.join(ACTIVE_DIR, 'image')):
        os.makedirs(os.path.join(ACTIVE_DIR, 'image'))
    if not os.path.exists(os.path.join(ACTIVE_DIR, 'masks')):
        os.makedirs(os.path.join(ACTIVE_DIR, 'masks'))

    date = datetime.datetime.now()
    now_str = date.strftime('%Y-%m-%d-%H%M%S')
    suffix = 'i{}xp{}xe{}_{}_{}_{}'.format(INIT_PICK,
                                           TRAIN_PICK,
                                           epochs,
                                           pick,
                                           stopping,
                                           now_str)

    # prepare models
    model_list = []
    for m in model:
        model_filename = '{}_{}'.format(m, suffix)
        _model = MODEL_MAPPING[m](input_size=INPUT_SIZE, loss=LOSS)

        model_list.append(AlModel(m, _model, model_filename, ))

    unpicked = glob.glob(os.path.join(TRAIN_DIR, 'image/*.png'))
    _ = pick_copy_images(unpicked, INIT_PICK, strategy=pick_strategy)
    n_train_samples = INIT_PICK

    train_round = 1
    while True:
        steps_per_epoch = (n_train_samples // BATCH_SIZE) * 3

        print('===== Starting round {} with {} images ====='.format(train_round,
                                                                    n_train_samples))

        # train model by model
        for m in model_list:
            train_epoch = 1
            while epoch_strategy(train_epoch, max_epochs=EPOCHS_PER_ROUND_CONSTANT):
                # refresh generator to prevent unsafe thread error
                my_data_gen = train_generator(BATCH_SIZE, ACTIVE_DIR,
                                              'image', 'masks',
                                              img_target_size=IMG_TARGET_SIZE,
                                              augs=data_gen_args)

                m.fit_generator(my_data_gen, steps_per_epoch=steps_per_epoch,
                                validation_data=(X_val, Y_val))
                train_epoch += 1

            if not unpicked or stop_strategy(histories=m.histories):
                m.finished = True

        if all([m.finished for m in model_list]):
            break

        # pick next batch of images
        _ = pick_copy_images(unpicked, TRAIN_PICK)
        n_train_samples += TRAIN_PICK
        train_round += 1

    plot_histories(model_list, suffix)

    # evaluate models
    print('Starting evaluation on test set...')
    evals = [m.eval() for m in model_list]

    print('========== STATS ==========')
    print('Model\t| loss\t| val_loss\t| acc\t| iou\t| val_acc\t| val_iou\t| test_acc\t| test_iou\t|')
    for m, e in zip(model_list, evals):
        print('{}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t' \
              '| {:.4f}\t| {:.4f}\t' \
              '| {:.4f}\t| {:.4f}\t|'.format(m.name,
                                             m.get_best('loss', greater_better=False),
                                             m.get_best('val_loss', greater_better=False),
                                             m.get_best('acc'),
                                             m.get_best('mean_iou'),
                                             m.get_best('val_acc'),
                                             m.get_best('val_mean_iou'),
                                             e[1],
                                             e[2]))


cli.add_command(simulate)


if __name__ == '__main__':
    cli()
