import datetime
import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.bib_detector import PreTrainedModel
from classes.bib_reader import OCRReader, OCRType
from classes.depth import ArrivalLine
from classes.person_detector import YOLOv11
from classes.pipeline import Pipeline
from classes.tools import crop, get_colored_logger

# Those are the parameters to set for the script to work
START_FRAME = 0  # The frame at which to start
END_FRAME = 0  # The frame number to end
PARAMETER_FILE = ""  # The file to read parameters from (for the cropping region and the arrival line decription)

# You can use one of the functions below to write the parameters to some presets ones


RESULT_FOLDER = "./results/runs"  # The folder to save the results to

ANNOTATE = True  # If you want to annotate the videos and save the annoted videos to a file
DETAIL_ANNOTATE = True  # If you want all the detailed vizu (depth picture, person detection, bib detection)

TIMESTAMP_CSV = "./data/recorded/merged/right_merged_full.csv"
SOURCE_VIDEO = "./data/recorded/merged/right_merged.mp4"


def set_first_clip():
    # use this function to set the global parameters for the video between 10min and 01h05
    global START_FRAME, END_FRAME, PARAMETER_FILE
    START_FRAME = 10 * 60 * 30  # 00:10:00
    END_FRAME = 75 * 60 * 30  # 01:05:00
    PARAMETER_FILE = "parameters/parameters_first_hour.json"
    pass


def set_second_clip():
    # use this function to set the global parameters for the video between 02h15 and 03h40
    global START_FRAME, END_FRAME, PARAMETER_FILE
    START_FRAME = 135 * 60 * 30  # 02:15:00
    END_FRAME = 220 * 60 * 30  # 03:40:00
    PARAMETER_FILE = "parameters/parameters_second_hour.json"
    pass


set_first_clip()

logging.basicConfig(level=logging.DEBUG)

logger = get_colored_logger(__name__)


if DETAIL_ANNOTATE and not ANNOTATE:
    logger.warning("If DETAIL_ANNOTATE is True, ANNOTATE needs to be true too. correcting")
    ANNOTATE = True

curr_path = ""


def find_result_path():
    global curr_path
    runs_index = 0
    while True:
        curr_path = os.path.join(RESULT_FOLDER, f"run_{runs_index}")
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
            break
        runs_index += 1


find_result_path()
logger.info(f"Processing will start. Output folder : {curr_path}")

parameters = {}


with open(PARAMETER_FILE) as file:
    parameters = json.loads("\n".join(file.readlines()))


cap = cv2.VideoCapture(SOURCE_VIDEO)
df = pd.read_csv(TIMESTAMP_CSV, header=None, names=["frame_n", "timestamp"], delimiter=";")


line_detector = ArrivalLine(line=parameters["line"])

pipeline = Pipeline(
    person_detector=YOLOv11("./models/base/yolo11s.pt"),
    bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
    bib_reader=OCRReader(type=OCRType.PADDLE),
    line=line_detector,
    annotate=ANNOTATE,
    detail_annotate=True,
)


width = parameters["crop"][1][0] - parameters["crop"][0][0]
height = parameters["crop"][1][1] - parameters["crop"][0][1]
if ANNOTATE:
    writer = cv2.VideoWriter(os.path.join(curr_path, "out.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(width, height))
if DETAIL_ANNOTATE:
    depth_writer = cv2.VideoWriter(os.path.join(curr_path, "out_depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(width, height))
    person_writer = cv2.VideoWriter(os.path.join(curr_path, "out_person.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(width, height))
    bib_writer = cv2.VideoWriter(os.path.join(curr_path, "out_bib.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(width, height))


def sequential_pipe():
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    for i in tqdm(range(START_FRAME, END_FRAME)):
        ret, frame = cap.read()
        if not ret:
            logger.info("End of recording reached")
            break
        frame = crop(frame, parameters["crop"])
        frames = pipeline.new_frame(frame, i, parralel=True)
        if ANNOTATE:
            writer.write(frames["annoted"])
        if DETAIL_ANNOTATE:
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            depth_writer.write(frames["depth"] if "depth" in frames else black_frame)
            person_writer.write(frames["person"] if "person" in frames else black_frame)
            bib_writer.write(frames["bib"] if "bib" in frames else black_frame)
        if i % 10000 == 0:
            logger.info(f"Done frame {i - START_FRAME}/{END_FRAME - START_FRAME}")


def batched_pipe(batch_size=10):
    # !!! This code does not work for now. Take it as an example of how I would do it.
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    for i in tqdm(range(START_FRAME, END_FRAME, batch_size)):
        frames = []
        for j in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                logger.info("End of recording reached")
                break
            frame = crop(frame, parameters["crop"])
            frames.append(frame)
        pipeline.new_frames(np.array(frames), list(range(i, i + len(frames))))


sequential_pipe()
if ANNOTATE:
    writer.release()

pipeline.clean_detections()
res = {
    "frame_start": START_FRAME,
    "frame_end": END_FRAME,
}
for id, p in pipeline.persons.items():
    time = -1
    if p.frame_passed_line != -1:
        time = float(df["timestamp"][p.frame_passed_line])
    curr_conf = -1
    if p.best_bib is not None:
        curr_conf = p.best_bib.curr_conf
    best_bib_text = ""
    if p.best_bib is not None:
        best_bib_text = p.best_bib.bib_text
    res[id] = {
        "time": time,
        "best_bib": best_bib_text,
        "passed_line": p.passed_line,
        "confidence": curr_conf,
        "bibs": [(b.curr_conf, b.bib_text) for b in p.bibs.values()],
        "video_timestamp": str(datetime.timedelta(seconds=p.last_detected / 30)),
    }


with open(os.path.join(curr_path, "results.json"), "w") as outfile:
    outfile.write(json.dumps(res))
