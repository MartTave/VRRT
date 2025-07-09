import concurrent.futures

import cv2
import numpy as np

from classes.bib_detector import BibDetector
from classes.bib_reader import BibReader
from classes.depth import ArrivalLine
from classes.person_detector import PersonDetector
from classes.tools import get_colored_logger

logger = get_colored_logger(__name__)


def check_bib_in_person(bib_box, person_boxes):
    bib_center_x = (bib_box[0] + bib_box[2]) / 2
    bib_center_y = (bib_box[1] + bib_box[3]) / 2

    if person_boxes.id is None:
        logger.warning("Did not have an id for tracking !")
        return False, None

    for i, p_box in enumerate(person_boxes.xyxy):
        if (p_box[0] <= bib_center_x <= p_box[2]) and (p_box[1] <= bib_center_y <= p_box[3]):
            return True, person_boxes.id[i]
    return False, None


def treat_bib_result(args):
    bib_box, bib_reader, frame, person_result = args
    res = check_bib_in_person(bib_box, person_result.boxes)
    if res[0]:
        person_id = int(res[1])
        if person_id is None:
            return None
        cropped_bib = frame[bib_box[1].int() : bib_box[3].int(), bib_box[0].int() : bib_box[2].int()].copy()

        res = bib_reader.read_frame(cropped_bib)
        if res is None:
            return None
        return (res[0], res[1], person_id)
    return None


def box_to_points(box):
    arr = box.numpy().astype(np.int32)
    return arr[:2], arr[2:]


class Bib:
    bib_text: str
    curr_conf: float = 0.0
    conf_tresh: float
    detected: bool = False

    def __init__(self, bib_text, other_bibs=[], conf_tresh=1.5):
        self.bib_text = bib_text
        # At the bib creation, we check if the new text contains any other bib text
        # If yes, we add the confidence of the others bib to this one
        for b in other_bibs:
            if b.bib_text in bib_text:
                self.curr_conf += b.curr_conf
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
    passed_line: bool
    frame_passed_line: int

    def __init__(self, id):
        self.id = id
        self.best_bib = None
        self.bibs = {}
        self.last_detected = 0
        self.passed_line = False
        self.frame_passed_line = -1

    def update_best_bib(self):
        for b in self.bibs.values():
            if self.best_bib is None or self.best_bib.curr_conf < b.curr_conf:
                self.best_bib = b

    def detected_bib(self, text: str, conf: float):
        if text not in self.bibs.keys():
            self.bibs[text] = Bib(text, other_bibs=self.bibs.values())
        for b in self.bibs.values():
            if text in b.bib_text:
                # We add the confidence to each bib that contains the detection, to allow for partial detections
                b.new_detection(conf)
        self.update_best_bib()


class Pipeline:
    person_detector: PersonDetector
    bib_detector: BibDetector
    bib_reader: BibReader
    persons: dict[int, Person] = {}

    def __init__(
        self,
        person_detector: PersonDetector,
        bib_detector: BibDetector,
        bib_reader: BibReader,
        line: ArrivalLine,
        annotate=False,
        detail_annotate=False,
    ):
        self.annotate = annotate
        self.detail_annotate = detail_annotate
        self.person_detector = person_detector
        self.bib_detector = bib_detector
        self.bib_reader = bib_reader
        self.persons = {}
        self.line = line
        # Number of frame that a person can be undetected before being treated as out of frame
        self.grace_not_detected = 300  # = 10 sec at 30FPS
        logger.info("Pipeline initialized !")

    def remove_useless_persons(self, current_frame_index):
        self.persons = {
            k: v
            for k, v in self.persons.items()
            if v.passed_line or len(v.bibs) > 0 or current_frame_index - v.last_detected < self.grace_not_detected
        }

    def keep_only_boxes(self, frame, boxes):
        mask = np.zeros_like(frame)
        if boxes.device != "cpu":
            boxes = boxes.cpu()
        for b in boxes:
            points = box_to_points(b)
            cv2.rectangle(mask, points[0], points[1], (255, 255, 255), thickness=cv2.FILLED)
        masked_image = frame.copy()
        masked_image = cv2.bitwise_and(masked_image, mask)
        return masked_image

    def treat_new_frame_result(self, frame, frame_index, person_result, bib_result, depth):
        frames = {"annoted": frame.copy()}

        if person_result is None or len(person_result.boxes.xyxy) == 0:
            # If no person are detected, we can't do anyhting...
            return frames

        arrived = self.line.treat_depth(depth, person_result, frame, self.annotate)

        for p_id in person_result.boxes.id:
            p_id = int(p_id)
            if p_id not in self.persons.keys():
                self.persons[p_id] = Person(p_id)
            if p_id in arrived:
                self.persons[p_id].passed_line = True
                self.persons[p_id].frame_passed_line = frame_index
            self.persons[p_id].last_detected = frame_index

        if self.detail_annotate:
            frames["person"] = self.keep_only_boxes(frame, person_result.boxes.xyxy)

        if bib_result is not None:
            if self.detail_annotate:
                frames["bib"] = self.keep_only_boxes(frame, bib_result.boxes.xyxy)
            for bib_box in bib_result.boxes.xyxy:
                res = check_bib_in_person(bib_box, person_result.boxes)
                if res[0]:
                    person_id = int(res[1])
                    if person_id is None:
                        continue
                    cropped_bib = frame[bib_box[1].int() : bib_box[3].int(), bib_box[0].int() : bib_box[2].int()].copy()

                    res = self.bib_reader.read_frame(cropped_bib)
                    if res is not None:
                        bib, confidence = res
                        self.persons[person_id].detected_bib(bib, confidence)
        if self.annotate:
            for box, id in zip(person_result.boxes.xyxy, person_result.boxes.id, strict=False):
                id = int(id)
                curr_pers = self.persons[id]
                color = (0, 0, 255)
                if curr_pers.passed_line:
                    color = (0, 255, 0)
                bib_text = "?"
                bib_color = (0, 0, 255)
                if curr_pers.best_bib is not None:
                    bib_text = f"'{curr_pers.best_bib.bib_text}'"
                    bib_color = (0, 255, 0)

                text_1 = f"Id : {id}"
                text_2 = f"Bib {bib_text}"
                # Draw box around person
                person_points = box_to_points(box)
                cv2.rectangle(frames["annoted"], person_points[0], person_points[1], color=color)

                # Draw text for person info
                cv2.putText(
                    frames["annoted"],
                    text_1,
                    (person_points[0][0], person_points[0][1] - 14),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 255),
                )
                cv2.putText(
                    frames["annoted"], text_2, person_points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=bib_color, thickness=1
                )
            if bib_result is not None:
                for box in bib_result.boxes.xyxy.cpu():
                    points = box_to_points(box)
                    cv2.rectangle(frames["annoted"], points[0], points[1], color=(255, 0, 255))
            cv2.line(frames["annoted"], self.line.line_points[0], self.line.line_points[1], (255, 255, 0), 3)
            if self.detail_annotate:
                new_frame = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                new_frame = new_frame.astype(np.uint8)
                new_frame = cv2.applyColorMap(new_frame, cv2.COLORMAP_JET)
                new_frame = cv2.line(new_frame, self.line.line_points[0], self.line.line_points[1], (255, 0, 255), 3)
                frames["depth"] = new_frame
        if frame_index % 10 == 0:
            self.remove_useless_persons(frame_index)
        return frames

    def new_frame(self, frame, frame_index, parralel=True):
        if parralel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all three methods to the executor
                future_person = executor.submit(self.person_detector.detect_persons, frame)
                future_bib = executor.submit(self.bib_detector.detect_bib, frame)
                future_depth = executor.submit(self.line.model.infer_image, frame)

                # Wait for all futures to complete and get results
                person_result = future_person.result()
                bib_result = future_bib.result()
                depth = future_depth.result()
        else:
            person_result = self.person_detector.detect_persons(frame)
            bib_result = self.bib_detector.detect_bib(frame)
            depth = self.line.model.infer_image(frame)
        return self.treat_new_frame_result(frame, frame_index, person_result, bib_result, depth)

    def new_frames(self, frames, frames_indexes):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all three methods to the executor
            future_person = executor.submit(self.person_detector.detect_persons_multiple, frames)
            future_bib = executor.submit(self.bib_detector.detect_bib_multiple, frames)
            future_depth = executor.submit(self.line.model.infer_images, frames)

            # Wait for all futures to complete and get results
            person_results = future_person.result()
            bib_results = future_bib.result()
            depths = future_depth.result()
        for frame, frame_index, person_result, bib_result, depth in zip(frames, frames_indexes, person_results, bib_results, depths, strict=True):
            self.treat_new_frame_result(frame, frame_index, person_result, bib_result, depth)

    def clean_detections(self):
        # We pass the biggest number available as int32 in order to flush every person detected that has not passed the line and have no bib detected
        self.remove_useless_persons(2_147_483_647)
