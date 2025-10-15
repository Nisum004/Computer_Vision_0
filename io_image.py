import os
import cv2
print(cv2.__version__)

# READ IMAGE
image_path = os.path.join('.' , 'data' , 'bird.jpeg')

img = cv2.imread(image_path)

# WRITE IMAGE

cv2.imwrite(os.path.join('.' , 'data' , 'bird_out.jpeg'), img)

# VISUALIZE IMAGE

cv2.imshow('image' , img)
cv2.waitKey(0)
