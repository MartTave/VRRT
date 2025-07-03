from abc import ABC, abstractmethod

import torch
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
    def __init__(self, model_path, device=torch.device(0)) -> None:
        self.model = YOLO(model_path)
        self.device = f"{device.type}:{device.index}" if device.index else f"{device.type}"

    @override
    def detect_bib(self, pic: MatLike) -> None:
        results = self.model(pic, verbose=False, device=self.device)
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
        results = self.model(pics, verbose=False, batch=len(pics))
        assert results is not None
        return results
