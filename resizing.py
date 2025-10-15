# resizing

import os
import cv2

img = cv2.imread(os.path.join('.','data','bird.jpeg'))

resized_img = cv2.resize(img, (640, 480))

print(img.shape)
print(resized_img.shape)

cv2.imshow('image' , img)
cv2.imshow('resized_image' , resized_img)
cv2.waitKey(0)
