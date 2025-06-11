import time

import cv2


def set_props(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 20)


cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(4)

for _ in range(30):
    _, _ = cap1.read()
    _, _ = cap2.read()

set_props(cap1)
set_props(cap2)

for _ in range(30):
    _, _ = cap1.read()
    _, _ = cap2.read()


frames1 = []

frames2 = []
start = time.time()
for i in range(30 * 1):
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()

    frames1.append(frame1)
    frames2.append(frame2)

print(f"FPS : {(30 * 30) / (time.time() - start)}")

for i in range(0, len(frames1), 10):
    stacked = cv2.hconcat((frames1[i], frames2[i]))
    cv2.imshow("frames", stacked)
    while True:
        key = cv2.waitKey()
        if key == ord("q"):
            break
