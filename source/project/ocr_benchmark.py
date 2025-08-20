from posix import wait
import uuid
import json
import logging
import os

from classes.bib_reader import OCRReader, OCRType
from classes.detectors import cropFromBoxes
import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


correct_read = 0
partial_read = 0
no_read = 0
wrong_read = 0

bib_detector = YOLO("./models/fine_tuned/best_20.pt")
easy_ocr = OCRReader(type=OCRType.EASYOCR, conf_treshold=0.7)
paddle_ocr = OCRReader(type=OCRType.PADDLE, conf_treshold=0.7)
dir = "./data/dataset/"
labels = {}
with open(dir + "labels.json") as file:
    lines = file.readlines()
    labels = json.loads("\n".join(lines))


def get_filename(index):
    return f"pic_{str(index).zfill(3)}.png"


def get_file_index(filename):
    index = str(int("".join(filename.split(".")[:-1]).split("_")[-1]))
    return index


files = list(os.listdir(dir))
files.sort()

found = 0
false_pos = 0
total = 0
total_detected = 0

for i, f in enumerate(files):
    if not f.endswith(".png"):
        continue
    filepath = dir + f
    label_index = get_file_index(f)
    ground_truth = labels[label_index]
    frame = cv2.imread(filepath)
    results = bib_detector(frame, verbose=False)
    # results[0].show()
    boxes = results[0].boxes
    if boxes is None:
        continue
    cropped = cropFromBoxes(frame, boxes)
    read = []
    for c in cropped:
        text = easy_ocr.read_frame(c)
        text_2 = paddle_ocr.read_frame(c)
        if text != text_2:
            if text is not None and text_2 is not None and text[0] == text_2[0]:
                continue
            cv2.imshow("test", c)
            print(f"For frame {i} : {text} - {text_2}")
            key = cv2.waitKey()
            if key == ord('s'):
                cv2.imwrite(f"{i}.png", c)
        if text is None:
            continue
        read.append(text)
    total += len(ground_truth)
    total_detected += len(boxes)
    if len(boxes) < len(ground_truth):
        print(f"Missing bib detection : {len(read)} - {len(ground_truth)}")
    for bib, conf in read:
        if bib in ground_truth:
            found += 1
        else:
            false_pos += 1
            print(f"False pos : {bib}, label is : {ground_truth}")

print(f"Acc is : {found / total}")
print(f"Total : {total}")
print(f"Total detected : {total_detected}")
print(f"False positive : {false_pos}")
print(f"found : {found}")
import ipdb;ipdb.set_trace()
