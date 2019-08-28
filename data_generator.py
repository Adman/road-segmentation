import glob
import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
from skimage.transform import resize
import cv2

Road = [128, 64, 128]
Noroad = [255, 73, 95]
COLORS = np.array([Road, Noroad])


def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def normalize_image(img, colorspace='rgb'):
    if colorspace == 'rgb':
        return img / 255
    elif colorspace == 'hsv':
        return img / [180, 255, 255]


def train_generator(batch_size, train_path, image_folder, mask_folder,
                    img_target_size=(480, 640), augs={}, tohsv=False):
    if tohsv:
        image_datagen = ImageDataGenerator(preprocessing_function=rgb_to_hsv,
                                           **augs)
    else:
        image_datagen = ImageDataGenerator(**augs)
    masks_datagen = ImageDataGenerator(**augs)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        class_mode=None,
        classes=[image_folder],
        color_mode='rgb',
        target_size=img_target_size,
        batch_size=batch_size,
        seed=1
    )

    mask_generator = masks_datagen.flow_from_directory(
        train_path,
        class_mode=None,
        classes=[mask_folder],
        color_mode='grayscale',
        target_size=img_target_size,
        batch_size=batch_size,
        seed=1
    )

    colorspace = 'rgb'
    if tohsv:
        colorspace = 'hsv'

    generator = zip(image_generator, mask_generator)

    #def _s(i, m, c):
    #    green = np.ones(i.shape, dtype=np.float) * (0, 255, 0)
    #    transparency = .25
    #    p = m / 255
    #    p *= transparency
    #    # green over original image
    #    out = green*p + (i)*(1.0-p)

    #    # save mask overlaying image
    #    io.imsave(os.path.join('data/xxx', '{}.png'.format(c)), out.astype(np.uint8))
    #
    #counter = 0
    for (img, mask) in generator:
    #    _s(img[0], mask[0], counter)
    #    counter += 1
    #    _s(img[1], mask[1], counter)
    #    counter += 1

        img = normalize_image(img, colorspace=colorspace)
        mask /= 255
        yield (img, mask)


def load_data_memory(train_paths, image_folder, mask_folder, resize=(640, 480),
                     tohsv=False):
    X = []
    Y = []

    colorspace = 'rgb'
    if tohsv:
        colospace = 'hsv'

    for train_path in train_paths:
        for i in glob.glob(os.path.join(train_path, image_folder, '*.png')):
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize)
            if tohsv:
                img = rgb_to_hsv(img)
            img = normalize_image(img, colorspace=colorspace)
            X.append(img)

        for i in glob.glob(os.path.join(train_path, mask_folder, '*.png')):
            mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, resize)
            mask = mask.reshape(resize[1], resize[0], 1)
            mask = mask / 255
            Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)


def test_data_generator(test_path, image_folder, img_target_size=(480, 640),
                        tohsv=False):
    augs = {}
    if tohsv:
        augs['preprocessing_function'] = rgb_to_hsv

    test_datagen = ImageDataGenerator(**augs)

    image_gen = test_datagen.flow_from_directory(
        test_path,
        class_mode=None,
        classes=[image_folder],
        target_size=img_target_size,
        batch_size=1
    )

    colorspace = 'rgb'
    if tohsv:
        colorspace = 'hsv'

    for img in image_gen:
        img = normalize_image(img, colorspace=colorspace)
        yield img


def eval_generator(batch_size, test_path, image_folder, mask_folder,
                   img_target_size=(480, 640), tohsv=False):
    return train_generator(batch_size, test_path, image_folder, mask_folder,
                           img_target_size, {}, tohsv=tohsv)


def save_predicted_images(path, test_image_folder, predictions, img_target_size):
    os.makedirs(path, exist_ok=True)
    test_imgs = sorted(glob.glob(os.path.join(test_image_folder, '*.png')))

    for p, t in zip(predictions, test_imgs):
        p[p <= 0.5] = 0
        p[p > 0.5] = 255

        img = io.imread(t)
        img = resize(img, img_target_size)
        basename = os.path.basename(t)

        # save predicted mask
        #io.imsave(os.path.join(path, 'mask_{}'.format(basename)), p)

        green = np.ones(img.shape, dtype=np.float) * (0, 1, 0)
        transparency = .25
        p /= 255
        p *= transparency
        # green over original image
        out = green*p + img*(1.0-p)

        # save mask overlaying image
        io.imsave(os.path.join(path, '{}'.format(basename)), out)
