from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Full paths to your resources
model_path = '/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/final_emotion_detection_model.keras'
cascade_path = '/Users/divagarvakeesan/PycharmProjects/MobileNet-Based Real-Time Emotion Recognition from Facial Expressions/haarcascade_frontalface_default.xml'

# Load the pre-trained model and cascade classifier
classifier = load_model(model_path)
face_classifier = cv2.CascadeClassifier(cascade_path)

# Print the model summary to check if it's loaded correctly
print(classifier.summary())

# Open the webcam to detect emotions
cap = cv2.VideoCapture(0)

# Class labels (update these to match the labels used during training)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)

        # Convert grayscale ROI to RGB
        roi_rgb = cv2.merge([roi_gray, roi_gray, roi_gray])
        roi_rgb = roi_rgb.astype('float') / 255.0
        roi_rgb = np.expand_dims(roi_rgb, axis=0)  # Add batch dimension

        # Make a prediction on the ROI
        preds = classifier.predict(roi_rgb)[0]
        label = class_labels[preds.argmax()]
        label_position = (x, y - 10)  # Position label above the face rectangle
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
