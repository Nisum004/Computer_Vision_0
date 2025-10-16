import os
import cv2
import numpy as np
img = cv2.imread(os.path.join('.','data','lamelo.jpeg'))

img_edge = cv2.Canny(img,100,200)

img_edge_d = cv2.dilate(img_edge, np.ones((3,3), dtype = np.int8)) # make the white border thicker
img_edge_e = cv2.erode(img_edge_d, np.ones((4,4), dtype = np.int8)) # make the white border thinner

cv2.imshow('erode', img_edge_e)
cv2.imshow('dilated', img_edge_d)
cv2.imshow('edge',img_edge)
cv2.imshow('img', img)

cv2.waitKey(0)