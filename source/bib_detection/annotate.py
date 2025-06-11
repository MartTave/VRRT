import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

model = YOLO("./models/yolo11n.pt")

BASE_FOLDER = "./data/roboflow/train"

image_dir = os.path.join(BASE_FOLDER, "images")
existing_label_dir = os.path.join(BASE_FOLDER, "labels")
output_label_dir = os.path.join(BASE_FOLDER, "labels_updated")

os.makedirs(output_label_dir, exist_ok=True)

index = 0

for img_name in tqdm(os.listdir(image_dir)):
    index += 1
    img_path = os.path.join(image_dir, img_name)
    results = model(img_path, classes=[0], conf=0.3, verbose=False)  # Class 0 = 'person'

    if index % 100 == 0:
        annoted_img = results[0].plot()
        cv2.imshow("frame", annoted_img)
        cv2.waitKey()

    # Read existing bib number labels (if any)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    existing_label_path = os.path.join(existing_label_dir, label_name)
    new_label_path = os.path.join(output_label_dir, label_name)

    # Copy existing bib annotations (class 1)
    new_lines = []
    if os.path.exists(existing_label_path):
        with open(existing_label_path, "r") as f:
            new_lines = f.readlines()  # Bib lines (class 1)

    # Add new person annotations (class 0)
    for box in results[0].boxes:
        x_center, y_center, width, height = box.xywhn[0].tolist()
        new_lines.append(f"0 {x_center} {y_center} {width} {height}\n")

    # Save merged labels
    with open(new_label_path, "w") as f:
        f.writelines(new_lines)
