import os
import cv2

from Blur import img_median_blur

img2 = cv2.imread(os.path.join('.','data','noise.jpeg'))

img_median_blur2 = cv2.medianBlur(img2, 5)

cv2.imshow('image', img2)
cv2.imshow('median_blur', img_median_blur2)

cv2.waitKey(0)