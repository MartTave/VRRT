from ultralytics.data.loaders import time
from classes.person_detector import PersonDetector
from classes.bib_reader import BibReader
from classes.bib_detector import BibDetector
import numpy as np
import cv2
from classes.tools import crop_from_boxes, get_colored_logger, annotate_box

logger = get_colored_logger(__name__)

class Pipeline:

    person_detector: PersonDetector
    bib_detector: BibDetector
    bib_reader: BibReader

    def __init__(self, person_detector:PersonDetector, bib_detector:BibDetector, bib_reader:BibReader):
        self.person_detector = person_detector
        self.bib_detector = bib_detector
        self.bib_reader = bib_reader
        logger.info("Pipeline initialized !")



    def new_frame(self, frame, debug=False):
        person_result  = self.person_detector.detect_persons(frame)
        if person_result is None:
            return frame
        annoted_frame = frame.copy()
        annoted_frame = annotate_box(annoted_frame, person_result.boxes)
        for i, pers in enumerate(person_result.boxes.xyxy):
            cropped_person = frame[pers[1].int():pers[3].int(), pers[0].int():pers[2].int()].copy()
            bib_result = self.bib_detector.detect_bib(cropped_person)
            if bib_result is not None:
                annoted_frame = annotate_box(annoted_frame, bib_result.boxes, basepoint=(pers[0], pers[1]), color=(0, 255, 0))
                for bib in bib_result.boxes.xyxy:
                    cropped_bib = cropped_person[bib[1].int():bib[3].int(), bib[0].int():bib[2].int()].copy()
                    bib_text = self.bib_reader.read_frame(cropped_bib)
                    if bib_text is not None:
                        pass

        return annoted_frame
