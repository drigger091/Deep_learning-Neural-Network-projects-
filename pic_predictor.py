import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def predict_mask(input_image_path, model):
    input_image = cv2.imread(input_image_path)
    input_image_resized = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resized / 255
    input_image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    if input_pred_label == 0:
        print('The person in the image is wearing a mask')
    else:
        print('The person in the image is not wearing a mask')

# Usage example
input_image_path = input('Path of the image to be predicted: ')
model_path = 'D:\Face_mask_detect\Trained_model.h5'  # Replace with the actual path to your trained model

model = tf.keras.models.load_model(model_path)

predict_mask(input_image_path, model)