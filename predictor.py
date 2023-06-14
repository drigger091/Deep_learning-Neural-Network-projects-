import cv2
import numpy as np
from tensorflow import keras


model = keras.models.load_model('trained_model.h5')

cap = cv2.VideoCapture(0)

class_labels = ['Mask', 'No Mask']

def detect_face_mask(img):

    y_pred = model.predict(img.reshape(1,224,224,3))
    return y_pred[0][0]

def draw_label(img,text ,pos,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)


while True:
    ret, frame = cap.read()
    img = cv2.resize(frame,(224,224))
    
   
    y_pred = detect_face_mask(img)

    if y_pred == 1:
        draw_label(frame,"Without Mask ",(30,30),(0,255,0))
    else:
        draw_label(frame,"  Mask ",(30,30),(0,0,255))
    
    
    cv2.imshow("window",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
