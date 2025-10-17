import cv2
import os
import numpy as np

def get_face_landmarks(image, face_mesh):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []
    if results.multi_face_landmarks:
        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_, ys_, zs_ = [], [], []

        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)

        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))

    return image_landmarks