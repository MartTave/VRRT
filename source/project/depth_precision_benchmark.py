import json
import os
import time
from glob import glob

import cv2
import torch
from tqdm import tqdm

from classes.bib_detector import PreTrainedModel
from classes.bib_reader import OCRReader, OCRType
from classes.depth import ArrivalLine
from classes.depth_anything_v2.dpt import DepthAnythingV2
from classes.person_detector import YOLOv11
from classes.pipeline import Pipeline


def generate_depth_speed_benchmark():
    def crop_bottom_right(image, new_width, new_height):
        height, width = image.shape[:2]
        x = width - new_width
        y = height - new_height
        return image[y:height, x:width]

    DEVICE = 0

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    frames = []
    files = sorted(list(glob("./data/dataset/pic_*.png")))[:50]
    for f in files:
        frame = cv2.imread(f)
        frames.append(crop_bottom_right(frame, 1280, 720))

    encoder = "vits"  # or 'vits', 'vitb', 'vitg'
    for encoder in ["vits", "vitb"]:
        for size in [(128, 72), (256, 144), (384, 216), (512, 288), (640, 360), (1280, 720)]:
            model = DepthAnythingV2(**model_configs[encoder])
            model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
            model = model.to(DEVICE).eval()

            images = []
            then1 = time.time()

            for f in frames:
                image, (h, w) = model.image2tensor(f)
                images.append((image, h, w))
            print("Tensor created !")
            then2 = time.time()
            depths = []
            for i in images:
                depths.append(model.infer_image_cuda(*i))

            print(f"{encoder} --- {size} : ")
            print(f"Done {len(depths)} in {time.time() - then2} avg : {len(depths) / (time.time() - then2)} FPS")

            for i, depth in enumerate(depths):
                if i == 27:
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = depth.astype(np.uint8)
                    frame = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                    cv2.imwrite(f"./test/{encoder}_{size[0]}x{size[1]}.png", frame)


def generate_depth_precision_benchmark():
    VIDEO_FILENAME = "depth_benchmark_1.mp4"

    TIMESTAMP_FILENAME = VIDEO_FILENAME.replace(".mp4", ".txt")

    LABELS_FILNAME = "runs_label.json"

    PARAMETERS_FILENAME = "depth_benchmark_1_parameters.json"

    BASE_FOLDER = "./data/depth_precision/"

    SOURCE_VIDEO = os.path.join(BASE_FOLDER, VIDEO_FILENAME)

    SOURCE_LABELS = os.path.join(BASE_FOLDER, LABELS_FILNAME)

    PARAMETERS = os.path.join(BASE_FOLDER, PARAMETERS_FILENAME)

    parameters = {}

    timestamps = []

    with open(os.path.join(BASE_FOLDER, TIMESTAMP_FILENAME), "r") as file:
        lines = file.readlines()
        for l in lines:
            if l != "":
                timestamp = float(l)
                timestamps.append(timestamp)

    def crop(frame, points):
        return frame[points[0][1] : points[1][1], points[0][0] : points[1][0]]

    with open(PARAMETERS) as file:
        parameters = json.loads("\n".join(file.readlines()))

    cap = cv2.VideoCapture(SOURCE_VIDEO)

    res = {}

    def get_closest_frame(timestamp, arr, found_index=0):
        if len(arr) == 1 or len(arr) == 0:
            return found_index
        curr_index = len(arr) // 2
        if arr[curr_index] > timestamp:
            return get_closest_frame(timestamp, arr[0:curr_index], found_index=found_index)
        elif arr[curr_index] < timestamp:
            return get_closest_frame(timestamp, arr[curr_index:-1], found_index=curr_index + found_index)
        else:
            return curr_index + found_index

    with open(SOURCE_LABELS) as file:
        labels = json.load(file)

    width = parameters["crop"][1][0] - parameters["crop"][0][0]
    height = parameters["crop"][1][1] - parameters["crop"][0][1]

    serie_keys = ["body", "foot"]

    diffs = {}
    for serie_key in serie_keys:
        diffs[serie_key] = []
        for key, value in labels[serie_key].items():
            writer = cv2.VideoWriter(f"{BASE_FOLDER}{serie_key}/{serie_key}_{key}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, value["frame_start"])
            line_detector = ArrivalLine(line=parameters["line"])
            pipeline = Pipeline(
                person_detector=YOLOv11("./models/base/yolo11s.pt"),
                bib_detector=PreTrainedModel("./models/fine_tuned/best.pt"),
                bib_reader=OCRReader(type=OCRType.PADDLE),
                line=line_detector,
            )
            closest_frame = get_closest_frame(value["timestamp"], timestamps)
            for i in tqdm(range(value["frame_start"], value["frame_end"])):
                ret, frame = cap.read()
                frame = crop(frame, parameters["crop"])

                if not ret:
                    break
                pipeline.new_frame(frame, i, annotate=True, parralel=False)
                color = (0, 0, 255)
                if i == closest_frame:
                    color = (0, 255, 0)

                cv2.circle(frame, (20, 20), 5, color, lineType=cv2.FILLED, thickness=-1)
                writer.write(frame)
                # Process frame here
            writer.release()

            persons = pipeline.persons[1]
            timestamp = timestamps[persons.frame_passed_line]
            diff = timestamp - value["timestamp"]
            diffs[serie_key].append(diff)
            print(f"For {serie_key} {key} diff is : {diff}")
    return diffs


def compare_label_to_video():
    outlier = ["98"]

    labels = json.load(open("./data/frame_labels/bib_time_label.json"))
    results = json.load(open("./results/runs/second_part/results.json"))
    results_parsed = {
        results[key]["best_bib"]: results[key]["time"]
        for key in results.keys()
        if key not in ["frame_start", "frame_end"] and results[key]["best_bib"] in labels.keys()
    }
    diffs = []
    mean = 0
    for key, value in labels.items():
        if key in outlier:
            continue
        computed_time = results_parsed[key]
        diffs.append(value - computed_time)
        mean += abs(diffs[-1])
    mean /= len(labels.keys())
    import ipdb

    ipdb.set_trace()


compare_label_to_video()
