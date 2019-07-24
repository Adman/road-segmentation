import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob
import skimage.io as io
from skimage.transform import resize
import os
import cv2


Road = [128,64,128]
Noroad = [255, 73, 95]
COLORS = np.array([Road, Noroad])


def fix_mask(mask):
    '''The given mask should contain only 0 or 255 values'''
    mask[mask <= 100] = 0.0
    mask[mask > 100] = 1.0
    return mask


# inspired by
# https://github.com/zhixuhao/unet/blob/master/data.py#L4
# and https://github.com/keras-team/keras/issues/3059#issuecomment-364787723
# and https://stackoverflow.com/a/42462830/1442465
def train_generator(batch_size, train_path, image_folder, mask_folder, img_target_size=(320, 240), augs={}):
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

    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img /= 255
        mask /= 255
        yield (img, mask)


def load_data_memory(train_path, image_folder, mask_folder, resize=(320, 240)):
    X = []
    for i in glob.glob(os.path.join(train_path, image_folder, '*.png')):
        img = cv2.imread(i) / 255
        img = cv2.resize(img, resize)
        X.append(img)
    
    Y = []
    for i in glob.glob(os.path.join(train_path, mask_folder, '*.png')):
        mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, resize)
        mask = mask.reshape(resize[1], resize[0], 1)
        mask = mask / 255
        Y.append(mask)
    
    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)


#def test_data_generator(test_path):
#    for i in glob.glob(os.path.join(test_path, '*.png'), recursive=False):
#        img = io.imread(i)
#        yield img
def test_data_generator(test_path, image_folder, img_target_size=(240, 320)):
    test_datagen = ImageDataGenerator()

    image_gen = test_datagen.flow_from_directory(
        test_path,
        class_mode=None,
        classes=[image_folder],
        target_size=img_target_size,
        batch_size=1)
    
    for img in image_gen:
        img /= 255
        yield img


def eval_generator(batch_size, test_path, image_folder, mask_folder, img_target_size=(320, 240)):
    return train_generator(batch_size, test_path, image_folder, mask_folder, img_target_size, {})


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
        io.imsave(os.path.join(path, 'mask_{}'.format(basename)), p)

        green = np.ones(img.shape, dtype=np.float) * (0,1,0)
        transparency = .25
        p /= 255
        p *= transparency
        # green over original image
        out = green*p + img*(1.0-p)

        # save mask overlaying image
        io.imsave(os.path.join(path, '{}'.format(basename)), out)
