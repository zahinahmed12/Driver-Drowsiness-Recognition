import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(4)
model = tf.keras.models.load_model('./resnet50_eye_model_2')


def prediction_haarcascade():
    y_true = []
    y_pred = []

    path = './dataset_new/test/ny/'
    for i in os.listdir(path):

        img = cv2.imread(path + i)
        eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyess = eyeCascade.detectMultiScale(roi_gray)
            if len(eyess) == 0:
                print("eyes not detected")
            else:
                for ex, ey, ew, eh in eyess:
                    eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]
                    img = eyes_roi

            final_img = cv2.resize(img, (256, 256))
            final_img = np.expand_dims(final_img, axis=0)
            predict = model.predict(final_img)
            y_true.append(int(i.split('_')[0]))
            y_pred.append(1 if predict > 0.5 else 0)

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


def prediction_face(m):
    y_true = []
    y_pred = []
    # path = './dataset_B_FacialImages/test/'
    path = './dataset_new/test/ny/'

    for i in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, i), cv2.IMREAD_COLOR)
        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            resized_array = cv2.resize(roi_color, (256, 256))

            resized_array = np.array(resized_array)
            resized_array = np.expand_dims(resized_array, 0)

            y_true.append(int(i.split('_')[0]))
            y_pred.append(1 if m.predict(resized_array) > 0.5 else 0)

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


def prediction(m):
    y_true = []
    y_pred = []
    # path = './dataset_B_FacialImages/test/'
    path = './dataset_new/test/ny/'

    for i in os.listdir(path):
        img = Image.open(path + i)
        img = img.resize((256, 256))
        img = np.array(img)
        img = np.expand_dims(img, 0)

        y_true.append(int(i.split('_')[0]))
        y_pred.append(1 if m.predict(img) > 0.5 else 0)

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


def img_process():

    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    # img = tf.keras.utils.load_img('./dataset_new/test/ny/1_100.jpg', target_size=(256, 256))
    img = cv2.imread('./dataset_new/train/ny/0_112.jpg')
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # roi_grey = grey[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        img_array = cv2.resize(roi_color, (256, 256))
        # cv2.imshow('img', roi_color)
        # cv2.waitKey(0)
        return img_array


def main():

    # # img_array = tf.keras.utils.img_to_array(img)
    # # img = Image.open('./dataset_B_FacialImages/test/1_606.jpg')
    # img = Image.open('./dataset_new/train/ny/1_726.jpg')
    # img = img.resize((256, 256))
    # img_array = np.array(img)
    # # img_array = img_process()
    # img_array = tf.expand_dims(img_array, 0)  # Create a batch
    # # print(img_array.shape)
    # pred = model.predict(img_array)
    # print(prediction)
    # label = 1 if pred > 0.5 else 0
    # print(label)
    # # score = tf.nn.softmax(prediction[0])
    # # print(score)
    # # print(np.argmax(score), 100 * np.max(score))

    # prediction_face(model)
    # prediction(model)
    prediction_haarcascade()


if __name__ == '__main__':
    main()
    # img_process()
