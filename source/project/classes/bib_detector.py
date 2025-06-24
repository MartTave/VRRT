from abc import ABC, abstractmethod

from cv2.typing import MatLike
from overrides import override
from ultralytics import YOLO

from .tools import get_colored_logger

logger = get_colored_logger(__name__)


class BibDetector(ABC):
    @abstractmethod
    def detect_bib(self, pic: MatLike) -> None:
        pass

    @abstractmethod
    def detect_bib_multiple(self, pics: MatLike) -> list:
        pass


class PreTrainedModel(BibDetector):
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)

    @override
    def detect_bib(self, pic: MatLike) -> None:
        results = self.model(pic, verbose=False)
        if len(results) == 0:
            return None
        result = results[0]
        if result.boxes is None:
            return None
        elif len(result.boxes.xyxy) == 0:
            return None
        return result

    @override
    def detect_bib_multiple(self, pics: MatLike) -> list:
        results = self.model(pics, verbose=False)
        assert results is not None
        return results
