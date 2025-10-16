import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'whiteboard.jpg'))
print(img.shape)   # (width(x), height(y))

#l ine
cv2.line(img, (100,100), (512,390), (255,255,0), 5)

# rectangle
cv2.rectangle(img, (100,100), (512,390), (0,255), 2)   # -1 -> color filled rectangle

# circle
cv2.circle(img, (306,245), 100, (255, 0, 0), -1)

# Text
cv2.putText(img, 'Hello Nisum', (100,380), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,0,255), 2)

cv2.imshow('whiteboard.jpg', img)
cv2.waitKey(0)