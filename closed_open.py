import keras.optimizers
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications.vgg19 import VGG19    # For Transfer Learning
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import gradient_descent_v2
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout

np.random.seed(4)
batch_size = 1
epoch = 1
alpha = 0.001
lrr = ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=3, min_lr=1e-5)
sgd = gradient_descent_v2.SGD(learning_rate=alpha, momentum=.9, nesterov=False)
adam = adam_v2.Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


def closed_open_data():
    datagen = ImageDataGenerator()

    train_set = datagen.flow_from_directory('./dataset_new/train/co', class_mode='binary', batch_size=1234,
                                            shuffle=False)
    train_data = train_set[0][0]
    train_label = np.array(train_set[0][1]).astype(int)

    shuffled_train = np.random.permutation(train_data.shape[0])
    train_data = train_data[shuffled_train]
    train_label = train_label[shuffled_train]
    train_label = to_categorical(train_label)

    # train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    # train_generator = train_generator.flow(train_data, train_label, shuffle=False, sample_weight=None, batch_size=500)
    # train_generator.fit(train_data)

    test_set = datagen.flow_from_directory('./dataset_new/test/co', class_mode='binary', batch_size=218, shuffle=False)
    test_data = test_set[0][0]
    test_label = np.array(test_set[0][1]).astype(int)
    test_label = to_categorical(test_label)

    shuffled_test = np.random.permutation(test_data.shape[0])
    test_data = test_data[shuffled_test]
    test_label = test_label[shuffled_test]

    # test_generator = ImageDataGenerator(rescale=1/255)
    # test_generator = test_generator.flow(test_data, test_label, shuffle=False, sample_weight=None)
    # test_generator.fit(test_data)

    # print(len(test_generator))
    # print(len(train_generator))
    # print(train_generator[0])
    # print(train_data.shape)
    # print(train_label.shape)
    # print(len(train_label[train_label == 1]))
    # print(train_data[0:10])
    # print(train_label[300:])
    return train_data, train_label, test_data, test_label


def get_model():
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3), classes=2)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', input_dim=512))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # model.summary()
    return model


def main():

    x_train, y_train, x_tst, y_tst = closed_open_data()
    model = get_model()

    train_gen = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    train_gen.fit(x_train)
    tst_gen = ImageDataGenerator(rescale=1/255)
    tst_gen.fit(x_tst)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen.flow(x_train, y_train, batch_size=batch_size), epochs=epoch,
              steps_per_epoch=x_train.shape[0]//batch_size,
              validation_data=tst_gen.flow(x_tst, y_tst, batch_size=batch_size), validation_steps=250,
              callbacks=[lrr], verbose=1)


if __name__ == '__main__':
    main()
