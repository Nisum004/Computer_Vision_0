import os
import cv2

img = cv2.imread(os.path.join('.', 'data','bird.jpeg'))

#CLASSICAL BLUR
k_size = 11  # size of the neighborhood pixels ie proximity
img_blur = cv2.blur(img, (k_size,k_size))

#GAUSSIAN BLUR
img_gaussian_blur = cv2.GaussianBlur(img, (k_size,k_size), 5)

#MEDIAN BLUR
img_median_blur = cv2.medianBlur(img, k_size)

cv2.imshow('b_blur', img_median_blur)
cv2.imshow('gaussian_blur', img_gaussian_blur)
cv2.imshow('blur', img_blur)
cv2.imshow('image', img)

cv2.waitKey(0)