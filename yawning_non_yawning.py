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


# In[28]:


# rename files
# # get the files in the dataset_new/train/ny/yawn/ folder
# for i in os.listdir('./dataset_new/test/ny/yawn/'):
#     # get the path of the file
#     path = './dataset_new/test/ny/yawn/' + i
#     # get the name of the file
#     name = i.split('.')[0]
#     # get the path of the destination folder
#     dest = './dataset_new/test/ny/'
#     # # rename the file
#     os.rename(path, dest + '1_' + name + '.jpg')


# In[29]:


def prepare_df(data_type):
    path_link = "./dataset_new/" + data_type + "/ny"
    yaw_no = []
    IMG_SIZE = 256
    face_cas_path="./haarcascades/haarcascade_frontalface_default.xml"
    for image in os.listdir(path_link):
        class_num1 = image.split('_')[0]
        image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
        face_cascade = cv2.CascadeClassifier(face_cas_path)
        faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_color = img[y:y+h, x:x+w]
            resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
            yaw_no.append([resized_array, class_num1])
            cv2.imshow('image', img)
            cv2.waitKey(0)
    new_data = np.array(yaw_no)
    X = []
    y = []
    for feature, label in new_data:
        X.append(feature)
        y.append(label)
    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


def get_datagen():

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    return datagen


def get_model():

    feature_extractor = ResNet50(weights='imagenet',
                                 input_shape=(256, 256, 3),
                                 include_top=False)

    feature_extractor.trainable = False

    input_ = tf.keras.Input(shape=(256, 256, 3))

    x = feature_extractor(input_, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output_ = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(input_, output_)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def prediction(model):
    y_true = []
    y_pred = []

    for i in os.listdir('./dataset_new/test/ny/'):
        img = Image.open('./dataset_new/test/ny/' + i)
        img = img.resize((256, 256))
        img = np.array(img)
        img = np.expand_dims(img, 0)

        y_true.append(int(i.split('_')[0]))
        y_pred.append(1 if model.predict(img) > 0.5 else 0)

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


def main():
    X_train, y_train = prepare_df("train")
    X_test, y_test = prepare_df("test")
    train_datagen = get_datagen()
    test_datagen = get_datagen()

    # model = get_model()
    # model.fit(
    #     train_datagen.flow(X_train, y_train, batch_size=32),
    #     # steps_per_epoch=len(X_train) // 32,
    #     epochs=2)
    # prediction(model)


if __name__ == "__main__":
    main()
