import glob
import os

import cv2
import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Road = [128, 64, 128]
Noroad = [255, 73, 95]
COLORS = np.array([Road, Noroad])

aug_blurer = iaa.MotionBlur(k=(3, 7))
aug_fogger = iaa.Fog()
aug_brightness = iaa.imgcorruptlike.Brightness(severity=2)


def normalize_image(img, colorspace='rgb'):
    if colorspace == 'rgb':
        return img / 255
        # return img / 127.5 - 1
    elif colorspace == 'hsv':
        # cv2 rgb to hsv returns H value in range [0, 180] if the image
        # is of type int. In case of float, the range is [0, 360]
        img = img.astype(np.float32)
        if len(img.shape) == 4:
            for i in range(img.shape[0]):
                img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2HSV)
            c = img
        else:
            c = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return c / [360, 1, 255]


def fix_mask(mask):
    m = mask.copy()
    m[m < 30] = 0.0
    m[m > 0] = 1.0
    return m


def train_generator(batch_size, train_path, image_folder, mask_folder,
                    img_target_size=(480, 640), augs={}, tohsv=False,
                    aug=False):
    image_datagen = ImageDataGenerator(**augs)
    masks_datagen = ImageDataGenerator(preprocessing_function=fix_mask, **augs)

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

    for (img, mask) in generator:
        if aug:
            # blur augmentation
            img_aug = aug_blurer.augment_images(img)
            img_aug = normalize_image(img_aug, colorspace=colorspace)
            yield (img_aug, mask)

            # fog augmentation
            # img_aug = aug_fogger.augment_images(img)
            # img_aug = normalize_image(img_aug, colorspace=colorspace)
            # yield (img_aug, mask)

            # brightness augmentation
            img_aug = aug_brightness.augment_images(img.astype(np.uint8))
            img_aug = normalize_image(img_aug, colorspace=colorspace)
            yield (img_aug, mask)

        img = normalize_image(img, colorspace=colorspace)
        yield (img, mask)


def test_data_generator(test_path, image_folder, img_target_size=(480, 640),
                        tohsv=False):
    augs = {}
    test_datagen = ImageDataGenerator(**augs)

    image_gen = test_datagen.flow_from_directory(
        test_path,
        class_mode=None,
        classes=[image_folder],
        color_mode='rgb',
        target_size=img_target_size,
        batch_size=1,
        shuffle=False
    )

    colorspace = 'rgb'
    if tohsv:
        colorspace = 'hsv'

    for img in image_gen:
        img = normalize_image(img, colorspace=colorspace)
        yield (img,)


def eval_generator(batch_size, test_path, image_folder, mask_folder,
                   img_target_size=(480, 640), tohsv=False, aug=True):
    return train_generator(batch_size, test_path, image_folder, mask_folder,
                           img_target_size, augs={}, tohsv=tohsv, aug=aug)


def load_data_memory(train_paths, image_folder, mask_folder, resize=(640, 480),
                     tohsv=False, aug=True):
    X = []
    Y = []

    colorspace = 'rgb'
    if tohsv:
        colorspace = 'hsv'

    for train_path in train_paths:
        for i in sorted(glob.glob(os.path.join(train_path, image_folder, '*.png'))):
            img = cv2.imread(i).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize)

            if aug:
                img_aug = aug_blurer.augment_image(img)
                img_aug = normalize_image(img_aug, colorspace=colorspace)
                X.append(img_aug)

                # img_aug = aug_fogger.augment_images(img)
                # img_aug = normalize_image(img_aug, colorspace=colorspace)
                # X.append(img_aug)

                img_aug = aug_brightness.augment_image(img.astype(np.uint8))
                img_aug = normalize_image(img_aug, colorspace=colorspace)
                X.append(img_aug)

            img = normalize_image(img, colorspace=colorspace)
            X.append(img)

        for i in sorted(glob.glob(os.path.join(train_path, mask_folder, '*.png'))):
            mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, resize)
            mask = mask.reshape(resize[1], resize[0], 1)
            mask = fix_mask(mask)
            Y.append(mask)

            if aug:
                Y.append(mask)
                # Y.append(mask)
                Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)


def load_single_image(path, resize=(640, 480), tohsv=False):
    colorspace = 'rgb'
    if tohsv:
        colorspace = 'hsv'

    img = cv2.imread(path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize)
    img = normalize_image(img, colorspace=colorspace)
    return img


def save_predicted_images(path, test_image_folder, predictions, resize_to, save_mask):
    os.makedirs(path, exist_ok=True)
    test_imgs = sorted(glob.glob(os.path.join(test_image_folder, '*.png')))

    COLORIZED_ONES = np.ones((480, 640, 3)) * (1, 0, 1)

    for p, t in zip(predictions, test_imgs):
        p[p <= 0.5] = 0
        p[p > 0.5] = 255

        img = cv2.imread(t)
        img = cv2.resize(img, resize_to)
        basename = os.path.basename(t)

        if save_mask:
            # save predicted mask
            cv2.imwrite(os.path.join(path, 'mask_{}'.format(basename)), p)

        mask = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) * COLORIZED_ONES
        out = cv2.addWeighted(img, 1, mask.astype(np.uint8), 0.7, 0)

        cv2.imwrite(os.path.join(path, '{}'.format(basename)), out)


if __name__ == '__main__':
    data_gen_args = dict(horizontal_flip=True)
    gen = train_generator(2, 'data/train', 'image',
                          'masks',
                          augs=data_gen_args,
                          tohsv=False)
    for (img, mask) in gen:
        print(img)
