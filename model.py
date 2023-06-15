import tensorflow
from tensorflow import keras


def create_model():
    
    num_of_classes = 2

    model = keras.Sequential()
    
    try:

        model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


        model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.5))


        model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
        
        return model
    
    except ValueError as ve:
        print("Value Error occured:",str(ve))
    except TypeError as te:
        print("TypeError occured:",str(te))
    except Exception as e:
        print("An error occured:",str(e))

create_model()