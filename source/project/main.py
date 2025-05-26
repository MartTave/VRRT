import cv2
from detectors import Tracker, RunnerTracker, BibDetector

tracker = RunnerTracker(Tracker(), BibDetector())

cap = cv2.VideoCapture("./data/youtube/run_2.MP4")
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    tracker.new_frame(frame)
