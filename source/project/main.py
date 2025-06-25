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
from pipeline import Pipeline

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG)

logger = get_colored_logger(__name__)


pipeline = Pipeline(
    person_detector=YOLOv11("./models/base/yolo11s.pt"),
    bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
    bib_reader=OCRReader(type=OCRType.PADDLE),
)

cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
df = pd.read_csv("./data/recorded/merged/right_merged_full.csv", header=None, names=["frame_n", "timestamp"], delimiter=";")

logger.info("Recording started")
frame_limit = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_limit = 150_000


def sequential_pipe():
    for i in tqdm(range(0, frame_limit)):
        ret, frame = cap.read()
        if not ret:
            logger.info("End of recording reached")
            break
        pipeline.new_frame(frame, i)


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


try:
    sequential_pipe()
except Exception as e:
    print(f"Error was : {e}")


res = {}
for p in pipeline.persons.values():
    if len(p.bibs) > 0:
        new_bibs = []
        for b in p.bibs.values():
            curr_conf, curr_bib = b.curr_conf, b.bib_text
            new_conf = curr_conf
            for b2 in p.bibs.values():
                conf, bib = b2.curr_conf, b2.bib_text
                if bib != curr_bib and bib in curr_bib:
                    new_conf += conf
            new_bibs.append((new_conf, curr_bib))
        new_bibs.sort(key=lambda x: x[0], reverse=True)
        res[new_bibs[0][1]] = {
            "time": float(df["timestamp"][p.last_detected]),
            "confidence": new_bibs[0][0],
            "bibs": new_bibs,
            "video_timestamp": str(datetime.timedelta(seconds=p.last_detected / 30)),
        }


with open("results.json", "w") as outfile:
    outfile.write(json.dumps(res))
