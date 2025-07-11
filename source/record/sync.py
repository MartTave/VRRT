import asyncio
import time

import cv2


async def read_frame(cap):
    _, frame = cap.retrieve()
    return frame


def set_props(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(0)

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


async def reading():
    start = time.time()
    for i in range(30 * 5):
        before = time.time()
        # This is here to only grab the frame (not decoding them)
        # Allows for a better sync, but worse performance
        cap1.grab()
        cap2.grab()
        print(f"Took : {time.time() - before}")

        frame1, frame2 = await asyncio.gather(read_frame(cap1), read_frame(cap2))

        frames1.append(frame1)
        frames2.append(frame2)
    print(f"FPS : {(30 * 5) / (time.time() - start)}")


asyncio.run(reading())

for i in range(0, len(frames1), 10):
    stacked = cv2.hconcat((frames1[i], frames2[i]))
    cv2.imshow("frames", stacked)
    while True:
        key = cv2.waitKey()
        if key == ord("q"):
            break
