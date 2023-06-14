import os
import numpy as np
from keras.preprocessing import image
import cv2
import warnings
import random
warnings.filterwarnings("ignore")



def get_XY():
    categories = ['with_mask', 'without_mask']
    dataset = []

    for category in categories:
        path = os.path.join('data', category)
        label = categories.index(category)
        file_list = os.listdir(path)
        random.shuffle(file_list)  # Shuffle the file list

        count = 0  # Track the number of selected photos
        for file in file_list:
            if count >= 1200:
                break  # Stop iterating if we have reached the desired count

            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))

            dataset.append([img, label])
            count += 1

    random.shuffle(dataset)

    X = []
    Y = []

    for features, label in dataset:
        X.append(features)
        Y.append(label)

    X = np.array(X)
    X = X / 255  # scaling the X
    Y = np.array(Y)

    return X, Y
