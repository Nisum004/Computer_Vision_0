import os
import cv2
from util import get_face_landmarks
import numpy as np
import mediapipe as mp

data_dir = './emotion_dataset/train'
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  # static since reading images
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

output = []
for emotion_index, emotion in enumerate(os.listdir(data_dir)):
    emotion_path = os.path.join(data_dir, emotion)
    print(f"Processing {emotion}...")
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        face_landmarks = get_face_landmarks(image, face_mesh)
        if len(face_landmarks) == 1404:
            face_landmarks.append(emotion_index)
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))
print("✅ Saved features to data.txt")

# Clean up resources
face_mesh.close()