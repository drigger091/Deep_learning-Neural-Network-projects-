import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import warnings
warnings.filterwarnings("ignore")



def get_XY():

    with_mask_files = os.listdir('E:\GITHUB\Deep_learning-Neural-Network-projects-\data\with_mask')
    without_mask_files = os.listdir('E:\GITHUB\Deep_learning-Neural-Network-projects-\data\without_mask')
    with_mask_labels = [1]*len(with_mask_files)
    without_mask_labels = [0]*len(without_mask_files)

    labels = with_mask_labels + without_mask_labels

    with_mask_path = 'E:\GITHUB\Deep_learning-Neural-Network-projects-\data\with_mask/'

    data = []

    for img_file in with_mask_files:

        image = Image.open(with_mask_path + img_file)
        image = image.resize((128,128))
        image = image.convert("RGB")
        image = np.array(image)
        data.append(image)



    without_mask_path = 'E:\GITHUB\Deep_learning-Neural-Network-projects-\data\without_mask/'



    for img_file in without_mask_files:

        image = Image.open(without_mask_path + img_file)
        image = image.resize((128,128))
        image = image.convert("RGB")
        image = np.array(image)
        data.append(image)


    X = np.array(data)
    Y = np.array(labels)


    return X ,Y


get_XY()