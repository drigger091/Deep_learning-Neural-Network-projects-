import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = 'D:\Face_mask_detect\Trained_model.h5'  # Replace with the actual path to your trained model
model = tf.keras.models.load_model(model_path)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform mask prediction on the image
def predict_mask(image):
    # Resize and preprocess the image
    input_image_resized = cv2.resize(image, (224, 224))
    input_image_scaled = input_image_resized / 255
    input_image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    # Perform mask prediction
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 0:
        return 'Mask Detected'
    else:
        return 'No Mask Detected'

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

# Create a pop-up window to display the webcam feed
cv2.namedWindow('Mask Detection')

# Process frames from the webcam feed
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find the largest face
    largest_face = None
    largest_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)

    # Process the largest face
    if largest_face is not None:
        x, y, w, h = largest_face

        # Zoom in on the face by adjusting the region of interest (ROI)
        roi_x = x - int(0.2 * w)
        roi_y = y - int(0.2 * h)
        roi_w = int(1.4 * w)
        roi_h = int(1.4 * h)
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        if roi.size != 0:  # Check if ROI is empty
            try:
                # Perform mask prediction on the ROI
                result = predict_mask(roi)

                # Draw bounding box and label on the face
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                cv2.putText(frame, result, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print("Error during prediction:", str(e))

    # Display the frame in the pop-up window
    cv2.imshow('Mask Detection', frame)

    # Break the loop when 'x' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
