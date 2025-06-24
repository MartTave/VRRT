import typing
from abc import ABC, abstractmethod
from typing import override

from cv2.typing import MatLike
from ultralytics import YOLO

from .tools import get_colored_logger

logger = get_colored_logger(__name__)


class PersonDetector(ABC):
    @abstractmethod
    def detect_persons(self, frame) -> tuple[list[MatLike], typing.Any] | None:
        pass


class YOLOv11(PersonDetector):
    def __init__(self, model):
        self.model = YOLO(model)

    @override
    def detect_persons(self, frame) -> tuple[list[MatLike], typing.Any] | None:
        results = self.model.track(frame, verbose=False, tracker="./trackers/botsort.yaml", persist=True)

        if len(results) == 0:
            return None
        result = results[0]

        if result.boxes is None:
            return None

        result = result[result.boxes.cls == 0]  # Then filter to just persons

        return result
