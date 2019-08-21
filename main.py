import glob
import os
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import loss
import augmentation as aug
from model import get_unet_nikitin, get_unet_256

height = 256
width = 256

max_epochs = 100
batch_size = 4


def generator(train_files, train_masks, batch_size):
    num_samples = len(train_files)
    while 1:
        shuffle(train_files)
        for offset in range(0, num_samples, batch_size):
            batch_train_files = train_files[offset:offset + batch_size]
            batch_train_masks = train_masks[offset:offset + batch_size]

            train_images = []
            labels = []
            for train_image, label in zip(batch_train_files, batch_train_masks):
                image = cv2.imread(train_image)
                image = cv2.resize(image, (width, height))

                measurement = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
                if (len(measurement.shape) > 2):
                    measurement = measurement[:, :, 1]
                measurement = cv2.resize(measurement, (width, height))
                image = aug.randomHueSaturationValue(image,
                                                     hue_shift_limit=(-50, 50),
                                                     sat_shift_limit=(-5, 5),
                                                     val_shift_limit=(-15, 15))
                image, measurement = aug.randomShiftScaleRotate(image, measurement,
                                                                shift_limit=(-0.0625, 0.0625),
                                                                scale_limit=(-0.1, 0.1),
                                                                rotate_limit=(-0, 0))
                image, measurement = aug.randomHorizontalFlip(image, measurement)
                measurement = np.expand_dims(measurement, axis=2)
                train_images.append(image)
                labels.append(measurement)

            X_train = np.array(train_images) / 255

            y_train = np.array(labels) / 255

            yield shuffle(X_train, y_train)


def train():
    train_files = os.path.join('/media/dnikitin/work/austrian/img',
                               '*')
    train_masks = os.path.join('/media/dnikitin/work/austrian/ann',
                               '*')
    train_files = sorted(glob.glob(train_files))
    train_masks = sorted(glob.glob(train_masks))
    model = get_unet_nikitin(height,width)
    model.summary()

    train_samples, validation_samples, train_masks, validation_masks = train_test_split(train_files, train_masks,
                                                                                        test_size=0.2, random_state=17)
    train_generator = generator(train_samples, train_masks, batch_size=batch_size)
    validation_generator = generator(validation_samples, validation_masks, batch_size=batch_size)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/media/dnikitin/work/BIS/PeopleMasking/logs', histogram_freq=0,
                                             write_graph=True, write_images=True)
    tbCallBack.set_model(model)
    callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1,
                                                            mode='min')
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint('unet_multiline.h5', monitor='val_loss', verbose=1,
                                                                save_best_only=True,
                                                                save_weights_only=False, mode='min')

    callback_reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4,
                                                                   verbose=1, mode='auto', cooldown=0, min_lr=0)
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_samples) / batch_size,
                                  epochs=max_epochs,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  validation_steps=len(validation_samples) / batch_size,
                                  callbacks=[callback_reduce_on_plateau, callback_model_checkpoint,
                                             callback_early_stopping])
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean s quared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    plot_model(model, to_file='model.png')


if __name__ == "__main__":
    train()
