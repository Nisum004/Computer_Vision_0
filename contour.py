import os
import cv2

print(cv2.__version__)

#import images
img2 = cv2.imread(os.path.join('.','data','whiteboard.jpg'))
img = cv2.imread(os.path.join('.', 'data','crows.jpg'))

#gray -> threshold -> contour
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 167, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>100:

        # draw contours on any image
        cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
        cv2.drawContours(img2, [cnt], -1, (0, 255, 0), 1)

        # calculate bounding rect
        x1, y1, w, h = cv2.boundingRect(cnt)

        # draw rectangle on each contours
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,0,0), 2)

cv2.imshow('thresh_inv', thresh)
cv2.imshow('crows.jpg', img)

cv2.imshow('img2', img2)
cv2.waitKey(0)