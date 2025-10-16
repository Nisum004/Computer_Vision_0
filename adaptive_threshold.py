import os
import cv2

img = cv2.imread(os.path.join('.', 'data','paper.jpg'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21,30)

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.imshow('thresh2', thresh2)

cv2.waitKey(0)