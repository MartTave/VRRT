import datetime
import json
import logging

import cv2
import pandas as pd
from tqdm import tqdm

from classes.bib_detector import PreTrainedModel
from classes.bib_reader import OCRReader, OCRType
from classes.person_detector import YOLOv11
from classes.tools import get_colored_logger
from depth import ArrivalLine, get_arrival_line
from pipeline import Pipeline

def crop(frame, points):
    return frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG)

logger = get_colored_logger(__name__)


parameters = {}

with open("parameters.json") as file:
    parameters = json.loads("\n".join(file.readlines()))


START_FRAME = 10 * 60 * 30

END_FRAME = 75 * 60 * 30

cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
df = pd.read_csv("./data/recorded/merged/right_merged_full.csv", header=None, names=["frame_n", "timestamp"], delimiter=";")


line_detector = ArrivalLine(line=parameters["line"])

pipeline = Pipeline(
    person_detector=YOLOv11("./models/base/yolo11s.pt"),
    bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
    bib_reader=OCRReader(type=OCRType.PADDLE),
    line=line_detector
)

logger.info("Recording started")
frame_limit = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_limit = 150_000

ANNOTATE = True

def sequential_pipe():
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    for i in tqdm(range(START_FRAME, END_FRAME)):
        ret, frame = cap.read()
        if not ret:
            logger.info("End of recording reached")
            break
        frame = crop(frame, parameters["crop"])
        pipeline.new_frame(frame, i, ANNOTATE)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)


def batched_pipe(batch_size=10):
    for i in tqdm(range(0, frame_limit, batch_size)):
        frames = []
        for j in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                logger.info("End of recording reached")
                break
            frames.append(frame)
        pipeline.new_frame(frames, list(range(i, i + batch_size)))


sequential_pipe()

res = {}
for id, p in pipeline.persons.items():
    time = -1
    if p.frame_passed_line != -1:
        time = float(df["timestamp"][p.frame_passed_line])
    res[id] = {
        "time": time,
        "best_bib": p.best_bib,
        "passed_line": p.passed_line,
        "confidence": p.best_bib.curr_conf,
        "bibs": [(b.curr_conf, b.bib_text) for b in p.bibs.values()],
        "video_timestamp": str(datetime.timedelta(seconds=p.last_detected / 30)),
    }


with open("results.json", "w") as outfile:
    outfile.write(json.dumps(res))
