import cv2
import os
from PIL import Image
from util import get_limits

blue = [255,0,0] # blue in BGR colorspace

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color = blue)
    mask = cv2.inRange(hsv_img, lowerLimit, upperLimit)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(
            frame,
            "Color: Blue",
            (x1, y1 - 10),  # place text above the rectangle
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # green text
            2
        )
        area = (x2 - x1) * (y2 - y1)
        cv2.putText(frame, f"Area: {area}", (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()