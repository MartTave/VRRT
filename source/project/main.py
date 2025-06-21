import json
import logging
import time

import cv2
import pandas as pd
from tqdm import tqdm

from classes.bib_detector import PreTrainedModel
from classes.bib_reader import OCRReader, OCRType
from classes.person_detector import YOLOv11
from classes.tools import get_colored_logger
from pipeline import Pipeline

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

logger = get_colored_logger(__name__)


pipeline = Pipeline(
    person_detector=YOLOv11("./models/base/yolo11s.pt"),
    bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
    bib_reader=OCRReader(type=OCRType.PADDLE),
)

cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
df = pd.read_csv("./data/recorded/merged/right_merged_full.csv", header=None, names=["frame_n", "timestamp"], delimiter=";")


global_frame_index = 0
frame_n = None
for i in range(100):
    ret, frame_n = cap.read()
    global_frame_index += 1
    assert ret

# cropping_region = click_and_crop(frame_n)

logger.info("Recording started")

already_detected = []
results = {}
for i in tqdm(range(0, 150_000)):
    ret, frame = cap.read()
    if not ret:
        logger.info("End of recording reached")
        break
    time1 = time.time()
    # frame = frame[cropping_region[1][0]:cropping_region[1][1], cropping_region[0][0]:cropping_region[0][1]]
    detected_bibs = pipeline.new_frame(frame)
    if detected_bibs is not None and len(detected_bibs) > 0:
        for detected_bib in detected_bibs:
            if detected_bib not in already_detected:
                already_detected.append(detected_bib)
                cv2.imwrite(f"./debug/res/{detected_bib}.png", frame)
                results[detected_bib] = float(df["timestamp"][global_frame_index])
        pass
    # print(f"Took : {time1 - time.time()}")
    global_frame_index += 1


with open("results.json", "w") as outfile:
    outfile.write(json.dumps(results))
