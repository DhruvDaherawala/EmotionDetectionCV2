import cv2
import numpy as np
import tensorflow as tf
face_detector = cv2.CascadeClassifier("C:/Users/dhruv/Downloads/Emotion_Detection_CNN-main/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")
emotion_classifier = tf.keras.models.load_model("C:/Users/dhruv/Desktop/College/Sem 4/DE/Team Infinite/model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))
        emotion = emotion_classifier.predict(face)
        emotion_index = np.argmax(emotion)
        emotion_label = emotion_labels[emotion_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Face Emotion Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()