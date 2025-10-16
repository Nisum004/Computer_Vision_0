import os
import cv2

img = cv2.imread(os.path.join('.', 'data','bird.jpeg'))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# BGR to RBG , red <-> blue switch red and blue
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow('image_grAY', img_gray)
cv2.imshow('image_rgb', img_rgb)
cv2.imshow('image', img)
cv2.imshow('image_hsv', img_hsv)

cv2.waitKey(0)

