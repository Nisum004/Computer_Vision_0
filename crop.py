import os
import cv2

img = cv2.imread(os.path.join('.', 'data','bird.jpeg'))
print(img.shape)

x_start,y_start,x_end,y_end = 1000,100,2700,2000
# x_start,x_end = width (left to right)
# y_start,y_end = height (top to bottom)
cropped_img = img[y_start:y_end, x_start:x_end]
print(cropped_img.shape)

cv2.imshow('image' , img)
cv2.imshow('cropped_image' , cropped_img)
cv2.waitKey(0)