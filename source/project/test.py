import time

import cv2
from tqdm import tqdm

from classes.person_detector import YOLOv11
from depth import ArrivalLine, get_arrival_line

detector = YOLOv11("./models/base/yolo12n.pt")


start_frame = 8200

cap = cv2.VideoCapture("/home/marttave/projects/Bachelor/source/project/data/recorded/right/video_14.06.2025_10:35:57.mp4")

ret, frame = cap.read()

assert ret
line = get_arrival_line(frame)

line_detector = ArrivalLine(line=line)


cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

passed_line_ids = []

frames = []

batch = 1

for i in tqdm(range(start_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), batch)):
    for j in range(batch):
        ret, frame = cap.read()
        if ret:
            then = time.time()
            line_detector.load_frame(frame)
            frames.append(frame)
    person_result = detector.detect_persons_multiple(frames)
    boxes = [p.boxes for p in person_result]
    then = time.time()
    results = line_detector.treat_loaded_frames(boxes)
    print(f"Took : {time.time() - then}")
    for r in results:
        if len(r) > 0:
            passed_line_ids += r
    frames = []

    # for id, xyxy in zip(person_result.boxes.id, person_result.boxes.xyxy, strict=False):
    #     frame = cv2.rectangle(
    #         frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color=(0, 255, 0) if id in passed_line_ids else (0, 0, 255)
    #     )
    #     pass
    # cv2.imshow("frame", frame)
    # cv2.waitKey(1)
