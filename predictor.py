import cv2
import numpy as np
from tensorflow import keras


model = keras.models.load_model('trained_model.h5')

video_Capture = cv2.VideoCapture(0)

class_labels = ['Mask', 'Without Mask']


while True:
    ret, frame = video_Capture.read()

    
    frame = cv2.resize(frame, (128, 128))

    frame = frame / 255.0  

    predictions = model.predict(np.expand_dims(frame, axis=0))
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_Capture.release()
cv2.destroyAllWindows()
