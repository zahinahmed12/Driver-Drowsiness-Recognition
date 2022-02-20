import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


np.random.seed(4)
img_size = 256
batch_size = 32
epoch = 10
# alpha = 0.001


def prepare_df(data_type):
    X = []
    y = []
    all_data = []
    path = './data_mix/' + data_type

    # X = np.array([np.array(Image.open(path + i)) for i in os.listdir(path)])
    # y = np.array(i.split('_')[0] for i in os.listdir(path))

    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i), cv2.IMREAD_COLOR)
        img_arr = cv2.resize(img, (img_size, img_size))
        # data = np.asarray(img)
        label = i.split('_')[0]
        all_data.append([img_arr, label])

        # X.append(data)
        # y.append(i.split('_')[0])

    for feature, label in all_data:
        X.append(feature)
        y.append(label)

    X = np.array(X, dtype='float32')
    y = np.array(y)

    return X, y


def get_datagen():

    datagen = ImageDataGenerator(
        # rescale=1/255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    return datagen


def get_model():

    feature_extractor = ResNet50(weights='imagenet',
                                 input_shape=(256, 256, 3),
                                 include_top=False,
                                 classes=4)

    feature_extractor.trainable = False

    input_ = tf.keras.Input(shape=(256, 256, 3))

    x = feature_extractor(input_, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output_ = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(input_, output_)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    return model


def prediction(model, x_tst, y_tst, lbl):
    y_pred = model.predict(x_tst)
    y_true = np.argmax(y_tst, axis=1)

    print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=lbl))
    print()
    print(confusion_matrix(y_true, np.argmax(y_pred, axis=1)))


def predict_one_img(model):
    filepath = './data_mix/test/1_779.jpg'
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # img_array = img_array / 255
    resized_array = cv2.resize(img_array, (256, 256))
    arr = resized_array.reshape(-1, 256, 256, 3)
    pred = model.predict([arr])
    print(pred)
    print(np.argmax(pred))


def plotter(model):
    f, ax = plt.subplots(2, 1)

    # Loss
    ax[0].plot(model.history.history['loss'], color='b', label='Training Loss')
    ax[0].plot(model.history.history['val_loss'], color='r', label='Validation Loss')

    # Accuracy
    ax[1].plot(model.history.history['accuracy'], color='b', label='Training  Accuracy')
    ax[1].plot(model.history.history['val_accuracy'], color='r', label='Validation Accuracy')


def main():

    x_train, y_train = prepare_df("train")
    x_valid, y_valid = prepare_df("valid")
    x_test, y_test = prepare_df("test")

    print(x_train.shape, x_valid.shape, x_test.shape)

    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    train_datagen = get_datagen()
    valid_datagen = get_datagen()
    test_datagen = get_datagen()

    train_datagen.fit(x_train)
    valid_datagen.fit(x_valid)
    test_datagen.fit(x_test)

    model = get_model()
    model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epoch,
              steps_per_epoch=x_train.shape[0] // batch_size, validation_data=valid_datagen.flow(
              x_valid, y_valid, batch_size=batch_size))

    # model.save('./resnet50_multiclass_model_mix')
    labels = ["Yawn", "Non_Yawn", "Closed", "Open"]
    prediction(model, x_test, y_test, labels)


if __name__ == '__main__':
    main()
