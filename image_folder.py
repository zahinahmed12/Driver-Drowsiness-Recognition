import os
import shutil
import numpy as np
import pandas as pd

np.random.seed(4)

# dir1 = './dataset_B_FacialImages/train/'
# if not os.path.isdir(dir1):
#     os.makedirs(dir1)

# dir2 = './dataset_B_FacialImages/valid/'
dir2 = './data/valid/no_yawn/'
if not os.path.isdir(dir2):
    os.makedirs(dir2)

dir3 = './data/test/no_yawn/'
# dir3 = './dataset_B_FacialImages/test'
if not os.path.isdir(dir3):
    os.makedirs(dir3)


def prepare_folder():
    X = []
    y = []
    # path1 = './dataset_B_FacialImages/' + '/co'
    path = './data/no_yawn/'
    for i in os.listdir(path):
        # Image
        X.append(i)
        # Label
        y.append(i.split('_')[0])

    X = np.array(X)
    y = np.array(y)

    df = pd.DataFrame()
    df['filename'] = X
    df['label'] = y

    # train, validate, test = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])
    validate, test = np.split(df.sample(frac=1), [int(.5 * len(df))])

    # train = np.array(train)
    validate = np.array(validate)
    test = np.array(test)

    print(validate.shape, test.shape)

    # for i in range(train.shape[0]):
    #
    #     shutil.copy('./dataset_B_FacialImages/co/'+train[i][0], dir1)
    for i in range(validate.shape[0]):

        # shutil.copy('./dataset_B_FacialImages/co/' + validate[i][0], dir2)
        shutil.copy(path + validate[i][0], dir2)
    for i in range(test.shape[0]):
        # os.path.join(dir3, test[i][0])
        # shutil.copy('./dataset_B_FacialImages/co/' + test[i][0], dir3)
        shutil.copy(path + test[i][0], dir3)
    return validate, test


prepare_folder()
