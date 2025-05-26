from abc import ABC, abstractmethod
from typing import override
from cv2.typing import MatLike
from ultralytics import YOLO

from .tools import crop_from_boxes, get_colored_logger

logger = get_colored_logger(__name__)

class BibDetector(ABC):

    @abstractmethod
    def detect_bib(self, pic:MatLike)->MatLike|None:
        pass


class PreTrainedModel(BibDetector):

    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    @override
    def detect_bib(self, pic:MatLike)->MatLike|None:
        results = self.model(pic, verbose=False)
        if len(results) == 0:
            return None
        result = results[0]
        if result.boxes is None:
            return None
        if len(result.boxes.xyxy) > 1:
            logger.debug("Got more than one bib detected for a person, ignoring...")
            return None
        elif len(result.boxes.xyxy) == 0:
            return None

        return result
