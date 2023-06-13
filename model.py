import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras import Sequential
from keras.layers import Dense


# building the convulational neural network


def create_model():

    vgg = VGG16()
    model = Sequential()
    try:
        for layer in vgg.layers[:-1]:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(1,activation='sigmoid'))

        return model
    
    except ValueError as ve:
        print("ValueError occurred:", str(ve))
    except TypeError as te:
        print("TypeError occurred:", str(te))
    except Exception as e:
        print("An error occurred:", str(e))



