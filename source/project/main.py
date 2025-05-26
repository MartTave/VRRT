import time
import cv2
from pipeline import Pipeline
from classes.bib_detector import PreTrainedModel
from classes.bib_reader import EasyOCR
from classes.person_detector import YOLOv11
from classes.tools import get_colored_logger, click_and_crop
import logging

logging.basicConfig(level=logging.INFO)

logger = get_colored_logger(__name__)


pipeline = Pipeline(person_detector=YOLOv11("./models/base/yolo11m.pt"), bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"), bib_reader=EasyOCR())

cap = cv2.VideoCapture("./data/youtube/run_2_high_quality.mp4")
frame_n = None
for i in range(1400):
    ret, frame_n = cap.read()
    assert ret

cropping_region = click_and_crop(frame_n)

logger.info("Recording started")

while True:
    time1 = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.info("End of recording reached")
        break
    time1 = time.time()
    frame = frame[cropping_region[1][0]:cropping_region[1][1], cropping_region[0][0]:cropping_region[0][1]]
    res_frame = pipeline.new_frame(frame, debug=True)
    cv2.imshow("frame", res_frame)
    cv2.waitKey(1)
