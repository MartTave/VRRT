import cv2 as cv
from record import get_capture
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", type=int, required=True, help="The input device id for the camera")
parser.add_argument("--fps", type=int, help="The FPS to which record the video")
parser.add_argument("-w", "--width", type=int, help="The width of the frame to capture", default=1920)
parser.add_argument("-h", "--height", type=int, help="The height of the frame to capture", default=1080)
args = parser.parse_args()

cap = get_capture(args.input, )

while True:
    ret, frame = cap.read()
    assert ret

    cv.imshow("frame", frame)
    cv.waitKey(1)
