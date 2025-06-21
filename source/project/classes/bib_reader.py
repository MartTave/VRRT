import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import override

import easyocr
from paddleocr import PaddleOCR

from .tools import get_colored_logger


class OCRType(Enum):
    PADDLE = 0
    EASYOCR = 1


logger = get_colored_logger(__name__)


class BibReader(ABC):
    @abstractmethod
    def read_frame(self, frame) -> str | None:
        pass


class OCRReader(BibReader):
    class Bib:
        bib_text: str
        curr_conf: float = 0.0
        conf_tresh: float
        detected: bool = False

        def __init__(self, bib_text, conf_tresh=1.5):
            self.bib_text = bib_text
            self.conf_tresh = conf_tresh

        def new_detection(self, conf):
            logger.debug(f"New detection for bib : {self.bib_text} at conf {self.curr_conf}")
            self.curr_conf += conf
            if self.curr_conf > self.conf_tresh and self.detected is False:
                self.detected = True
                logger.info(f"Valid new bib : {self.bib_text}")
                return self.bib_text

    conf_treshold: float
    accepted_chars: str
    bib_regx = re.compile("^[0-9]{1,4}([.][0-9]{1,3})?$")  # Match on <NUMBER> and <NUMBER.N>

    detected_bibs: dict[str, Bib] = {}

    def __init__(self, type: OCRType = OCRType.PADDLE, lang=["en"], conf_treshold=0.6, accepted_chars="1234567890.", bib_confidence_treshold=1.5):
        self.accepted_chars = accepted_chars
        self.conf_treshold = conf_treshold
        self.type = type
        if type == OCRType.EASYOCR:
            self.reader = easyocr.Reader(lang)
            self.readText = lambda x: self.reader.readtext(x)
        elif type == OCRType.PADDLE:
            self.reader = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            self.readText = lambda x: self.reader(x)[1]
        self.bib_confidence_treshold = bib_confidence_treshold

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
    def read_frame(self, frame) -> str | None:
        res = self.readText(frame)
        if len(res) == 0:
            logger.debug("No text detected in bib, ignoring")
            return None
        elif len(res) > 1:
            logger.debug("Multiple text detected for single bib, ignoring...")
            return None
        r = None
        if self.type == OCRType.PADDLE:
            r = ["", *res[0]]  # For Paddle OCR
        elif self.type == OCRType.EASYOCR:
            r = res[0]
        if r[2] < self.conf_treshold:
            logger.debug("Not high enough confidence, ignoring...")
            return None

        detected_text = self.bib_text_preprocess(r[1])
        bib_text = self.validate_bib_text(detected_text)

        if bib_text is False:
            logger.debug(f"Bib text is not valid : {r[1]} transformed into {detected_text}")
            return None

        if bib_text not in self.detected_bibs:
            self.detected_bibs[bib_text] = OCRReader.Bib(bib_text, self.bib_confidence_treshold)

        self.detected_bibs[bib_text].new_detection(r[2])

        if self.detected_bibs[bib_text].detected:
            return self.detected_bibs[bib_text].bib_text
