from ultralytics import YOLO
import cv2
import numpy as np

base_model = YOLO("yolo11m.pt")

bib_detection = YOLO("../bib_detection/weights/best_50.pt")


source_video = "../yolo/data/youtube/run_2.MP4"

cap = cv2.VideoCapture(source_video)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480))

    if not ret:
        break

    results_persons = base_model.track(frame, classes=0, verbose=False)

    results_bib = bib_detection(frame, verbose=False)

    annoted = results_persons[0].plot()
    annoted_2 = results_bib[0].plot()

    res = np.hstack((annoted, annoted_2))

    cv2.imshow("frame", res)
    cv2.waitKey(1)
