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



    def new_frame_backup(self, frame):
        person_result  = self.person_detector.detect_persons(frame)
        if person_result is None:
            return frame
        detected_bibs = []
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
                        detected_bibs.append(bib_text)
                        pass
        return detected_bibs

    def check_bib_in_person(self, bib_box, person_boxes):
        bib_center_x = (bib_box[0] + bib_box[2]) / 2
        bib_center_y = (bib_box[1] + bib_box[3]) / 2

        for p_box in person_boxes:
            if (p_box[0] <= bib_center_x <= p_box[2]) and (p_box[1] <= bib_center_y <= p_box[3]):
                return True
        return False

    def new_frame(self, frame):
        person_result  = self.person_detector.detect_persons(frame)
        if person_result is None:
            return []
        bib_result = self.bib_detector.detect_bib(frame)
        if bib_result is None:
            return []
        detected_bibs = []

        for bib_box in bib_result.boxes.xyxy:
            if self.check_bib_in_person(bib_box, person_result.boxes.xyxy):
                cropped_bib = frame[bib_box[1].int():bib_box[3].int(), bib_box[0].int():bib_box[2].int()].copy()
                bib_text = self.bib_reader.read_frame(cropped_bib)
                if bib_text is not None:
                    detected_bibs.append(bib_text)
                    pass
        return detected_bibs
