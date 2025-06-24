import json

import pandas as pd
from easyocr.DBNet.DBNet import cv2

results_computed = {}
results = {}

frames = []

base_df = pd.read_csv("./data/recorded/merged/right_merged_full.csv", delimiter=";", names=["frame_n", "timestamp"])

df = base_df["timestamp"].array

with open("./results.json", "r") as file:
    results_computed = json.loads("\n".join(file.readlines()))

with open("./data/race_results/parsed.json", "r") as file:
    results = json.loads("\n".join(file.readlines()))


times = [v["time"] for v in results_computed.values()]
confidences = [v["confidence"] for v in results_computed.values()]


def get_closest_frame(timestamp, arr=df, found_index=0):
    if len(arr) == 1 or len(arr) == 0:
        return found_index + 100
    curr_index = len(arr) // 2
    print(f"Curr index is : {curr_index}")
    if arr[curr_index] > timestamp:
        return get_closest_frame(timestamp, arr[0:curr_index], found_index=found_index)
    elif arr[curr_index] < timestamp:
        return get_closest_frame(timestamp, arr[curr_index:-1], found_index=curr_index + found_index)
    else:
        return curr_index + found_index + 100


def extract_video(offical_time, bib_text, found=False):
    start = offical_time + 40
    end = offical_time + 120
    frame_start = get_closest_frame(start)
    frame_end = get_closest_frame(end)
    if frame_start == frame_end:
        return
    print(f"Frame start : {frame_start} end : {frame_end}")
    cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    print(f"Total frames : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    writer = cv2.VideoWriter(f"./debug/bib/{bib_text}_{'found' if found else 'not_found'}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for i in range(frame_start, frame_end):
        ret, frame = cap.read()
        if not ret:
            print("Broke !")
            break
        writer.write(frame)
    writer.release()
    cap.release()


max_time = max(times) + 60

min_time = min(times)
print(min_time)
print(max_time)
bib_found = 0
bib_not_found = 0
count = 0
for key in results_computed.keys():
    if "." in key:
        continue
    if key not in results.keys():
        print(f"{key} not found in bib list")
        count += 1
print(f"{count} bib not found in list")

total = 0
total_found = 0
max_allowed_diff = 60.0
mean_diff = 0
max_diff = 0, 0
min_diff = 10000, 0
for key in results.keys():
    if results[key] > max_time:
        break
    if key in results_computed.keys():
        bib_found += 1
        diff = abs(results_computed[key]["time"] - results[key])
        if min_diff[0] > diff:
            min_diff = diff, key
        elif max_diff[0] < diff:
            max_diff = diff, key
        mean_diff += diff
        total_found += 1
        # extract_video(results[key], key, found=True)
    else:
        bib_not_found += 1
        # extract_video(results[key], key)

    total += 1


mean_diff /= total_found

print(f"Min diff: {min_diff} - max diff : {max_diff}")
print(f"Found {bib_found} of {total} bibs")
print(f"Mean diff is : {mean_diff}")
extract_video(1749888808.113506, "2161", found=True)
import ipdb

ipdb.set_trace()
