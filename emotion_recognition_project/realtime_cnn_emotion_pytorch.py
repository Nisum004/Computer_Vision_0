import cv2
import torch
import os
import numpy as np
from train_cnn_emotion_pytorch import CNNEmotion, device

# ---------------------------
# Load trained model
# ---------------------------
data_dir = './emotion_dataset/train'
emotions = sorted(os.listdir(data_dir))
num_classes = len(emotions)

model = CNNEmotion(num_classes).to(device)
model.load_state_dict(torch.load('../saved/cnn_emotion_pytorch.pth', map_location=device))
model.eval()

# ---------------------------
# Initialize webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# Real-time emotion recognition
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # flip horizontally
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,48,48)
        face_tensor = face_tensor.to(device) / 255.0  # normalize

        with torch.no_grad():
            output = model(face_tensor)
            pred_idx = output.argmax(1).item()
            emotion_text = emotions[pred_idx]

        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-time Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
