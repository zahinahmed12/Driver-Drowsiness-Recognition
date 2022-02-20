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
from classify import prepare_df
import matplotlib.pyplot as plt


def prediction(model, x_tst, y_tst, lbl):

    y_pred = model.predict(x_tst)
    y_true = np.argmax(y_tst, axis=1)

    print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=lbl))
    print()
    print(confusion_matrix(y_true, np.argmax(y_pred, axis=1)))


def predict_one_img(model):
    filepath = './data/test/all/1_113.jpg'
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
    ax[0].plot(model.history['loss'], color='b', label='Training Loss')
    ax[0].plot(model.history['val_loss'], color='r', label='Validation Loss')

    # Accuracy
    ax[1].plot(model.history['accuracy'], color='b', label='Training  Accuracy')
    ax[1].plot(model.history['val_accuracy'], color='r', label='Validation Accuracy')


def main():
    model = tf.keras.models.load_model('./resnet50_multiclass_model')
    predict_one_img(model)
    # plotter(model)
    # predict_one_img(model)
    # labels = ["Yawn", "Non_Yawn", "Closed", "Open"]
    # x_test, y_test = prepare_df("test")
    # y_test = to_categorical(y_test)
    # prediction(model, x_test, y_test, labels)


if __name__ == '__main__':
    main()
