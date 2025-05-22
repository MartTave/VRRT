from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2

class Tracker:

    def __init__(self, model="./models/yolo11n.pt", tracker_file="./trackers/botsort.yaml", verbose=False, classes=0,):
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
