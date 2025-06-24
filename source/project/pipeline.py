from classes.bib_detector import BibDetector
from classes.bib_reader import BibReader
from classes.person_detector import PersonDetector
from classes.tools import get_colored_logger

logger = get_colored_logger(__name__)


class Bib:
    bib_text: str
    curr_conf: float = 0.0
    conf_tresh: float
    detected: bool = False

    def __init__(self, bib_text, conf_tresh=1.5):
        self.bib_text = bib_text
        self.conf_tresh = conf_tresh
        self.last_detected = None

    def new_detection(self, conf):
        logger.debug(f"New detection for bib : {self.bib_text} at conf {self.curr_conf}")
        self.curr_conf += conf
        if self.curr_conf > self.conf_tresh and self.detected is False:
            self.detected = True
            logger.info(f"Valid new bib : {self.bib_text}")
            return self.bib_text


class Person:
    last_detected: int
    bibs: dict[str, Bib]
    best_bib: Bib | None

    def __init__(self, id):
        self.id = id
        self.best_bib = None
        self.bibs = {}
        self.last_detected = 0

    def update_best_bib(self):
        for b in self.bibs.values():
            if self.best_bib is None or self.best_bib.curr_conf < b.curr_conf:
                self.best_bib = b

    def detected_bib(self, text: str, conf: float):
        if text not in self.bibs.keys():
            self.bibs[text] = Bib(text)
        self.bibs[text].new_detection(conf)
        self.update_best_bib()


class Pipeline:
    person_detector: PersonDetector
    bib_detector: BibDetector
    bib_reader: BibReader
    persons: dict[int, Person] = {}

    def __init__(self, person_detector: PersonDetector, bib_detector: BibDetector, bib_reader: BibReader):
        self.person_detector = person_detector
        self.bib_detector = bib_detector
        self.bib_reader = bib_reader
        logger.info("Pipeline initialized !")

    def check_bib_in_person(self, bib_box, person_boxes):
        bib_center_x = (bib_box[0] + bib_box[2]) / 2
        bib_center_y = (bib_box[1] + bib_box[3]) / 2

        if person_boxes.id is None:
            logger.warning("Did not have an id for tracking !")
            return False, None

        for i, p_box in enumerate(person_boxes.xyxy):
            if (p_box[0] <= bib_center_x <= p_box[2]) and (p_box[1] <= bib_center_y <= p_box[3]):
                return True, person_boxes.id[i]
        return False, None

    def new_frame(self, frame, frame_index):
        person_result = self.person_detector.detect_persons(frame)

        if person_result is None or len(person_result.boxes.xyxy) == 0:
            return []
        bib_result = self.bib_detector.detect_bib(frame)
        if bib_result is None:
            return []
        detected_persons = []

        for bib_box in bib_result.boxes.xyxy:
            res = self.check_bib_in_person(bib_box, person_result.boxes)
            if res[0]:
                person_id = int(res[1])
                if person_id is None:
                    continue
                cropped_bib = frame[bib_box[1].int() : bib_box[3].int(), bib_box[0].int() : bib_box[2].int()].copy()
                if person_id not in self.persons.keys():
                    self.persons[person_id] = Person(person_id)
                self.persons[person_id].last_detected = frame_index
                res = self.bib_reader.read_frame(cropped_bib)
                if res is not None:
                    bib, confidence = res
                    self.persons[person_id].detected_bib(bib, confidence)

        return self.persons

    def new_frames(self, frames, frames_indexes):
        person_results = self.person_detector.detect_persons_multiple(frames)
        bib_results = self.bib_detector.detect_bib_multiple(frames)
        for frame, index, person_res, bib_res in zip(frames, frames_indexes, person_results, bib_results, strict=False):
            for bib_box in bib_res.boxes.xyxy:
                res = self.check_bib_in_person(bib_box, person_res.boxes)
                if res[0]:
                    person_id = int(res[1])
                    if person_id is None:
                        continue
                    cropped_bib = frame[bib_box[1].int() : bib_box[3].int(), bib_box[0].int() : bib_box[2].int()].copy()
                    if person_id not in self.persons.keys():
                        self.persons[person_id] = Person(person_id)
                    self.persons[person_id].last_detected = index
                    res = self.bib_reader.read_frame(cropped_bib)
                    if res is not None:
                        bib, confidence = res
                        self.persons[person_id].detected_bib(bib, confidence)
