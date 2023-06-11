import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('trained_model.h5')

# Setup webcam
video_Capture = cv2.VideoCapture(0)

# Class labels
class_labels = ['Mask', 'Without Mask']

# Process the webcam frames
while True:
    ret, frame = video_Capture.read()

    # Resize the frame
    frame = cv2.resize(frame, (128, 128))

    # Preprocess the frame
    frame = frame / 255.0  # Normalize the pixel values

    # Perform prediction
    predictions = model.predict(np.expand_dims(frame, axis=0))
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Display the predicted class label on the frame
    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_Capture.release()
cv2.destroyAllWindows()
