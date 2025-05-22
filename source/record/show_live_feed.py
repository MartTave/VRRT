import cv2 as cv
from record import get_capture

cap = get_capture(2)

while True:
    ret, frame = cap.read()
    assert ret

    cv.imshow("frame", frame)
    cv.waitKey(1)
