import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(4)


def prepare_df(data_type):
    X = []
    y = []
    path1 = './dataset_new/' + data_type + '/co'
    for i in os.listdir(path1):
        # Image
        X.append(i)
        # Label
        y.append(i.split('_')[0])

    X = np.array(X)
    y = np.array(y)

    df = pd.DataFrame()
    df['filename'] = X
    df['label'] = y

    return df


def get_datagen():

    datagen = ImageDataGenerator(
        # rescale=1/255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    return datagen


def get_data_flow(datagen, df, folder):
    path = './dataset_new/' + folder + '/co/'
    d_generator = datagen.flow_from_dataframe(
        df,
        directory=path,
        x_col='filename',
        y_col='label',
        class_mode='binary',
        target_size=(256, 256),
    )
    return d_generator


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

    # model.summary()
    return model


def prediction(model):
    y_true = []
    y_pred = []
    path = './dataset_new/test/co/'
    # path = './dataset_new/test/ny/'
    for i in os.listdir(path):
        img = Image.open(path + i)
        img = img.resize((256, 256))
        img = np.array(img)
        img = np.expand_dims(img, 0)

        y_true.append(int(i.split('_')[0]))
        y_pred.append(1 if model.predict(img) > 0.5 else 0)

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


def main():
    df_train = prepare_df("train")
    df_test = prepare_df("test")
    train_datagen = get_datagen()
    test_datagen = get_datagen()
    train_generator = get_data_flow(train_datagen, df_train, "train")
    test_generator = get_data_flow(test_datagen, df_test, "test")

    model = get_model()
    model.fit(train_generator, epochs=10)
    # model.save('./resnet50_eye_model')
    prediction(model)

    # img = tf.keras.utils.load_img('./dataset_new/train/ny/no_yawn/56.jpg', target_size=(256, 256))
    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)  # Create a batch
    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])
    # print(np.argmax(score), 100 * np.max(score))


if __name__ == '__main__':
    main()
