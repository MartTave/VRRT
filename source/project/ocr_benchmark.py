import json
import logging
import os

from classes.bib_reader import OCRReader, OCRType
from detectors import cropFromBoxes
from easyocr.detection import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


bib_detector = YOLO("./models/fine_tuned/best.pt")
ocr = OCRReader(type=OCRType.EASYOCR, conf_treshold=0.5, bib_confidence_treshold=0.5)
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

for f in files:
    if not f.endswith(".png"):
        continue
    filepath = dir + f
    label_index = get_file_index(f)
    ground_truth = labels[label_index]
    frame = cv2.imread(filepath)
    results = bib_detector(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        continue
    cropped = cropFromBoxes(frame, boxes)
    read = []
    for c in cropped:
        text = ocr.read_frame(c)
        if text is None:
            continue
        read.append(text)
    total += len(ground_truth)
    for r in read:
        if r in ground_truth:
            found += 1
        else:
            false_pos += 1

print(f"Acc is : {found / total}")
print(f"Total : {total}")
print(f"False positive : {false_pos}")
print(f"found : {found}")
