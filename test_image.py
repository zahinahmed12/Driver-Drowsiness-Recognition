import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import cv2

np.random.seed(4)
model = tf.keras.models.load_model('./resnet50_eye_model_2')


def img_process():

    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    # img = tf.keras.utils.load_img('./dataset_new/test/ny/1_100.jpg', target_size=(256, 256))
    img = cv2.imread('./dataset_new/train/ny/0_51.jpg')
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # roi_grey = grey[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        img_array = cv2.resize(roi_color, (256, 256))
        cv2.imshow('img', roi_color)
        cv2.waitKey(0)
        return img_array


def main():
    print("x")
    # img_array = tf.keras.utils.img_to_array(img)
    img_array = img_process()
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    print(predictions)
    # score = tf.nn.softmax(predictions[0])
    # print(score)
    # print(np.argmax(score), 100 * np.max(score))


if __name__ == '__main__':
    main()
    # img_process()
