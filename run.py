from data_generator import *
from models import *
from matplotlib import pyplot as plt


IMG_TARGET_SIZE = (240, 320)
RESIZE_TO = tuple(reversed(IMG_TARGET_SIZE))
INPUT_SIZE = (240, 320, 3)
BATCH_SIZE = 15
N_TRAIN_SAMPLES = len(glob.glob('data/train/image/*.png', recursive=False))
N_TEST_SAMPLES = len(glob.glob('data/test/image/*.png', recursive=False))


#data_gen_args = dict(zoom_range=0.05, horizontal_flip=True)
my_data_gen = train_generator(BATCH_SIZE, 'data/train', 'image', 'masks', img_target_size=IMG_TARGET_SIZE,
                              augs={})
val_data_gen = train_generator(1, 'data/val', 'image', 'masks', img_target_size=IMG_TARGET_SIZE)

steps_per_epoch = N_TRAIN_SAMPLES // BATCH_SIZE


# ------------------------------ choose model
#model = unet(pretrained_weights='unet_zajac.hdf5', input_size=INPUT_SIZE)
#model = unet(input_size=INPUT_SIZE)
model = FCN_Vgg16_32s(input_size=INPUT_SIZE)

model_checkpoint = ModelCheckpoint('unet_zajac.hdf5', monitor='loss', verbose=1, save_best_only=True)

#----------------- fit using generator
history = model.fit_generator(my_data_gen, steps_per_epoch, epochs=3,
                              callbacks=[model_checkpoint],
                              validation_data=val_data_gen, validation_steps=N_TRAIN_SAMPLES*0.1)


#----------------- fit by images in memory
#X_train, Y_train = load_data_memory('data/train', 'image', 'masks',resize=(320,240))
#history = model.fit(X_train, Y_train, epochs=3, batch_size=BATCH_SIZE, callbacks=[model_checkpoint],
#                    validation_split=0.1)


# ------------------------------- plot loss and accuracy
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend(loc='best')
plt.savefig('plots/loss.png')

plt.figure()
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend(loc='best')
plt.savefig('plots/acc.png')


# -------------------------------- evaluate model on test set
eval_gen = eval_generator(1, 'data/test', 'image', 'masks', img_target_size=IMG_TARGET_SIZE)
loss, acc = model.evaluate_generator(eval_gen, steps=N_TEST_SAMPLES, verbose=0)
print('Eval loss, acc: ', loss, acc)


# -------------------------------- predict image
test_gen = test_data_generator('data/test', 'image', img_target_size=IMG_TARGET_SIZE)
results = model.predict_generator(test_gen, steps=N_TEST_SAMPLES, verbose=1)
save_predicted_images('data/predictions', 'data/test/image', results, IMG_TARGET_SIZE)
