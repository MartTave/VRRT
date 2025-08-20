import typing
from abc import ABC, abstractmethod
from collections import defaultdict

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from overrides import override
from ultralytics import YOLO

from .tools import get_colored_logger

logger = get_colored_logger(__name__)


writter = cv2.VideoWriter("test_viz.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1617, 722))


class PersonDetector(ABC):
    @abstractmethod
    def detect_persons(self, frame) -> tuple[list[MatLike], typing.Any] | None:
        pass

    @abstractmethod
    def detect_persons_multiple(self, frames) -> list[tuple[list[MatLike], typing.Any] | None]:
        pass


track_history = defaultdict(lambda: [])


class YOLOv11(PersonDetector):
    def __init__(self, model, device=torch.device(0)):
        self.model = YOLO(
            model,
        )
        self.device = f"{device.type}:{device.index}" if device.index else f"{device.type}"

    @override
    def detect_persons(self, frame) -> tuple[list[MatLike], typing.Any] | None:
        results = self.model.track(frame, verbose=False, tracker="./trackers/botsort.yaml", persist=True, device=self.device, classes=[0])

        if len(results) == 0:
            return None
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return None

        # Visualize the result on the frame
        frame = result.plot()

        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids, strict=False):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y) + h / 2))  # x, y center point
            if len(track) > 120:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 0), thickness=5)

        writter.write(frame)
        # result = result[result.boxes.cls == 0]  # Filter to just persons

        return result

    @override
    def detect_persons_multiple(self, frames) -> list[tuple[list[MatLike], typing.Any] | None]:
        results = self.model.track(frames, verbose=False, tracker="./trackers/botsort.yaml", batch=len(frames), persist=True, imgsz=(384, 224))
        for r in results:
            r = r[r.boxes.cls == 0]
        return results
