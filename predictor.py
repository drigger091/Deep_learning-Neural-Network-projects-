import cv2
import numpy as np
from tensorflow import keras
import pickle


with open("training_history.pkl",'rb') as file:

    model = pickle.load(file)


# setup webcam

video_Capture = cv2.VideoCapture(0)


#class labels
class_labels = ['Mask','Without Mask']

#process the webcam frames

while True:

    ret , frame = video_Capture.read()

    # resize the frame
    frame = cv2.resize(frame,(128,128))

    normalized_frame = frame/255.0

    rgb_frame = cv2.cvtColor(normalized_frame,cv2.COLOR_BGR2RGB)

    input_frame = np.expand_dims(rgb_frame,axis=0)

    predictions = model.predict(input_frame)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]


    cv2.putText(frame,predicted_class_label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("webcam",frame)


    if cv2.waitKey(1)  == ord('q'):

        break


video_Capture.release()
cv2.destroyAllWindows()
