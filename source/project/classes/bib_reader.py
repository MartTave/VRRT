import logging
import re
from abc import ABC, abstractmethod
from enum import Enum

import torch
from overrides import override

from .tools import get_colored_logger


class OCRType(Enum):
    PADDLE = 0
    EASYOCR = 1


logger = get_colored_logger(__name__)


class BibReader(ABC):
    @abstractmethod
    def read_frame(
        self,
        frame,
    ) -> tuple[str, float] | None:
        pass


class OCRReader(BibReader):
    conf_treshold: float
    accepted_chars: str
    bib_regx = re.compile("^[0-9]{1,4}([.][0-9]{1,3})?$")  # Match on <NUMBER> and <NUMBER.N>

    def __init__(self, type: OCRType = OCRType.PADDLE, device=torch.device(0), lang=["en"], conf_treshold=0.6, accepted_chars="1234567890."):
        self.accepted_chars = accepted_chars
        self.conf_treshold = conf_treshold
        self.device = f"{device.type}:{device.index}" if device.index else f"{device.type}"
        self.type = type
        if type == OCRType.EASYOCR:
            import easyocr

            self.reader = easyocr.Reader(lang)
            self.readText = lambda x: self.reader.readtext(x)
        elif type == OCRType.PADDLE:
            from paddleocr import PaddleOCR

            logging.getLogger("ppocr").setLevel(logging.ERROR)

            self.reader = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
                device=self.device,
            )
            self.readText = lambda x: self.reader(x)[1]

    def bib_text_preprocess(self, bib_txt: str):
        bib_txt = bib_txt.strip()
        result = ""
        for c in bib_txt:
            if c in self.accepted_chars:
                result += c
        return result

    def validate_bib_text(self, bib_text) -> str | bool:
        match = re.match(self.bib_regx, bib_text)
        if match is None:
            return False

        return bib_text

    @override
    def read_frame(self, frame) -> tuple[str, float] | None:
        res = self.readText(frame)
        if len(res) == 0:
            logger.debug("No text detected in bib, ignoring")
            return None
        valid_texts = []
        for r in res:
            if self.type == OCRType.PADDLE:
                r = ["", *r]  # For Paddle OCR
            if r[2] < self.conf_treshold:
                logger.debug("Not high enough confidence, ignoring...")
                continue

            detected_text = self.bib_text_preprocess(r[1])
            text = self.validate_bib_text(detected_text)

            if text is False:
                logger.debug(f"Bib text is not valid : {r[1]} transformed into {detected_text}")
                continue
            valid_texts.append((text, r[2]))
        if len(valid_texts) > 1:
            logger.debug("More than one valid text detected, ignoring")
            return None
        elif len(valid_texts) == 0:
            logger.debug("No valid text detected, ignoring")
            return None
        valid = valid_texts[0]
        return valid
