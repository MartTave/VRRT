from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import easyocr
import uuid
import re

custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789."

reader = easyocr.Reader(["en"])

def cropFromBoxes(frame, boxes):
    res = []
    for b in boxes.xyxy:
        res.append(frame[b[1].int():b[3].int(), b[0].int():b[2].int()].copy())
    return res

class Tracker:

    def __init__(self, model="./models/base/yolo11n.pt", tracker_file="./trackers/botsort.yaml", verbose=False, classes=0,):
        self.model = YOLO(model)
        self.verbose = verbose
        self.tracker_file = tracker_file
        self.classes = classes
        self.track_history = defaultdict(lambda: [])


    def track(self, frame):
        results = self.model.track(frame, persist=True, classes=self.classes, verbose=False, tracker=self.tracker_file)
        return results

    def anotate(self, frame, results):
        annotated_frame = results[0].plot()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
        return annotated_frame


class BibDetector:

    def __init__(self, model="./models/fine_tuned/best.pt") -> None:
        self.model = YOLO(model)


    def infer(self, frame):
        return self.model(frame, verbose=False)


class RunnerTracker:

    bibs = {}
    bib_regx = re.compile("^[0-9]+(\.[0-9])?$") # Match on <NUMBER> and <NUMBER.N>

    def __init__(self, person_tracker:Tracker, bib_detector: BibDetector, bib_treshold=1.5) -> None:
        self.person_tracker = person_tracker
        self.bib_detector = bib_detector
        self.bib_treshold = bib_treshold

    def new_frame(self, frame, anotate=False):
        person_result = self.person_tracker.track(frame)
        boxes = person_result[0].boxes
        if boxes is not None:
            if boxes.id is None:
                return
            ids = boxes.id.int()
            persons_cropped = cropFromBoxes(person_result[0].orig_img, boxes)
            for i, p in enumerate(persons_cropped):
                p_id = ids[i]
                bib_detected = self.bib_detector.infer(p)
                bib_boxes = bib_detected[0].boxes
                if bib_boxes is not None:
                    bib_cropped = cropFromBoxes(p, bib_boxes)
                    for bib in bib_cropped:
                        res = reader.readtext(bib)
                        if len(res) > 1:
                            print("Detected more than one text in this image..")
                            cv2.imwrite(f"./debug/mult/{str(uuid.uuid4())}.jpg", bib)
                            continue
                        elif len(res) == 0:
                            cv2.imwrite(f"./debug/not_detected/{str(uuid.uuid4())}.jpg", bib)
                            continue
                        res = res[0]
                        if res[2] < 0.2:
                            print("Confidence is not high enough !")
                            cv2.imwrite(f"./debug/conf/{str(uuid.uuid4())}.jpg", bib)
                            continue
                        bib_n = res[1]

                        match = re.match(self.bib_regx, bib_n)
                        if match is None:
                            print(f"Detected a bib number that does not match the regex : {bib_n}")
                            continue

                        if bib_n not in self.bibs:
                            self.bibs[bib_n] = {
                            "confidence": 0,
                            "p_ids": [],
                            "accepted": False
                            }
                        self.bibs[bib_n]["confidence"] += res[2]
                        print(f"{bib_n} conf : {self.bibs[bib_n]['confidence']}")
                        if self.bibs[bib_n]["confidence"] >= self.bib_treshold and not self.bibs[bib_n]["accepted"]:
                            self.bibs[bib_n]["accepted"] = True
                            print(f"Found new bib : {bib_n}")
                            cv2.imwrite(f"./debug/res/{bib_n}.jpg", person_result[0].orig_img)
                        if p_id not in self.bibs[bib_n]["p_ids"]:
                            self.bibs[bib_n]["p_ids"].append(p_id)
