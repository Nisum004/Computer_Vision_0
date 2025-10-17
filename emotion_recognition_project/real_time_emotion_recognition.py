import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# -------------------------------
# Load saved model and scaler
# -------------------------------
model_path = "../saved/emotion_model.pkl"
scaler_path = "../saved/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or Scaler not found. Train the model first!")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------------
# Initialize Mediapipe FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# -------------------------------
# Define emotion labels (same order as your dataset folders)
# -------------------------------
emotion_labels = os.listdir('./emotion_dataset/train')
emotion_labels.sort()

# -------------------------------
# Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("🎥 Real-time Emotion Recognition started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Extract normalized landmark coordinates
            xs, ys, zs = [], [], []
            for lm in face_landmarks.landmark:
                xs.append(lm.x)
                ys.append(lm.y)
                zs.append(lm.z)

            feature_vector = []
            for j in range(len(xs)):
                feature_vector.append(xs[j] - min(xs))
                feature_vector.append(ys[j] - min(ys))
                feature_vector.append(zs[j] - min(zs))

            # Predict emotion if feature size is correct
            if len(feature_vector) == 1404:
                X = np.array(feature_vector).reshape(1, -1)
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]

                emotion_text = emotion_labels[int(pred)]
                cv2.putText(frame, f"Emotion: {emotion_text}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
