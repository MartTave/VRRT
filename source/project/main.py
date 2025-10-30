import json
import logging
import os
import time

import cv2
from classes.bib_detector import PreTrainedModel
from classes.bib_reader import OCRReader, OCRType
from classes.depth import ArrivalLine
from classes.person_detector import YOLOv11
from classes.pipeline import Pipeline
from classes.tools import get_colored_logger

# Those are the parameters to set for the script to work
START_FRAME = 0  # The frame at which to start
END_FRAME = 0  # The frame number to end
PARAMETER_FILE = "./parameters/parameters_demo.json"  # The file to read parameters from (for the cropping region and the arrival line decription)

# You can use one of the functions below to write the parameters to some presets ones


RESULT_FOLDER = "./results/runs"  # The folder to save the results to

ANNOTATE = True  # If you want to annotate the videos and save the annoted videos to a file
DETAIL_ANNOTATE = True  # If you want all the detailed vizu (depth picture, person detection, bib detection)

TIMESTAMP_CSV = "./data/recorded/merged/right_merged.csv"
SOURCE_VIDEO = 2


def set_first_clip():
    # use this function to set the global parameters for the video between 10min and 01h05
    global START_FRAME, END_FRAME, PARAMETER_FILE
    START_FRAME = 10 * 60 * 30  # 00:10:00
    START_FRAME = 34 * 60 * 30 + 20 * 30
    END_FRAME = 75 * 60 * 30  # 01:05:00
    END_FRAME = START_FRAME + 7 * 30
    PARAMETER_FILE = "parameters/parameters_first_hour.json"
    pass


def set_second_clip():
    # use this function to set the global parameters for the video between 02h15 and 03h40
    global START_FRAME, END_FRAME, PARAMETER_FILE
    START_FRAME = 135 * 60 * 30  # 02:15:00
    END_FRAME = 220 * 60 * 30  # 03:40:00
    PARAMETER_FILE = "parameters/parameters_second_hour.json"
    pass


# set_first_clip()

logging.basicConfig(level=logging.DEBUG)

logger = get_colored_logger(__name__)


if DETAIL_ANNOTATE and not ANNOTATE:
    logger.warning(
        "If DETAIL_ANNOTATE is True, ANNOTATE needs to be true too. correcting"
    )
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


parameters = {}


def get_capture(id, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


with open(PARAMETER_FILE) as file:
    parameters = json.loads("\n".join(file.readlines()))


cap = get_capture(4, 1920, 1080, 30)


line_detector = ArrivalLine(line=parameters["line"])

pipeline = Pipeline(
    person_detector=YOLOv11("./models/base/yolo11s.pt"),
    bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
    bib_reader=OCRReader(type=OCRType.PADDLE),
    line=line_detector,
    annotate=ANNOTATE,
    detail_annotate=False,
)


width = parameters["crop"][1][0] - parameters["crop"][0][0]
height = parameters["crop"][1][1] - parameters["crop"][0][1]


def sequential_pipe():
    i = 0
    then = time.time()
    while True:
        if i % 100 == 0:
            now = time.time()
            elapsed = now - then
            then = now
            print(f"FPS : {(100 / elapsed):.2f}")

        ret, frame = cap.read()
        if not ret:
            logger.info("End of recording reached")
            break
        frames = pipeline.new_frame(frame, i, parralel=False)
        cv2.imshow("Frame", frames["annoted"])
        cv2.waitKey(1)
        i += 1


sequential_pipe()
