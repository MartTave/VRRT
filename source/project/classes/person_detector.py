from abc import ABC, abstractmethod
from typing import override
import typing

from cv2.typing import MatLike
from ultralytics import YOLO


from .tools import crop_from_boxes, get_colored_logger

logger = get_colored_logger(__name__)

class PersonDetector(ABC):

    @abstractmethod
    def detect_persons(self, frame)->tuple[list[MatLike], typing.Any]|None:
        pass


class YOLOv11(PersonDetector):

    def __init__(self, model):
        self.model = YOLO(model)


    @override
    def detect_persons(self, frame) -> tuple[list[MatLike], typing.Any]|None:
        results = self.model.track(frame, classes=0, verbose=False, tracker="./trackers/botsort.yaml")

        if len(results) == 0:
            return None
        result = results[0]

        if result.boxes is None:
            return None


        return result
