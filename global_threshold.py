import os
import cv2

img = cv2.imread(os.path.join('.', 'data','gtrR34.jpg'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 26, 255, cv2.THRESH_BINARY)

thresh_blur = cv2.blur(thresh, (4,4))
ret, thresh = cv2.threshold(thresh_blur, 30, 255, cv2.THRESH_BINARY)


cv2.imshow('median_blur', thresh_blur)
cv2.imshow('thresh', thresh)
cv2.imshow('original', img)
cv2.waitKey(0)
