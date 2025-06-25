import cv2
from tqdm import tqdm

from classes.person_detector import YOLOv11
from depth import ArrivalLine

detector = YOLOv11("./models/base/yolo11s.pt")

line_detector = ArrivalLine()

cap = cv2.VideoCapture("./test.mp4")

for i in tqdm(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, frame = cap.read()

    if not ret:
        break

    person_result = detector.detect_persons(frame)
    pic = person_result.plot()
    cv2.imshow("frame", pic)
    cv2.waitKey(1)
    if person_result is not None and person_result.boxes is not None and person_result.boxes.id is not None:
        passed_line = line_detector.new_frame(frame, person_result.boxes)
        if len(passed_line) > 0:
            import ipdb

            ipdb.set_trace()
