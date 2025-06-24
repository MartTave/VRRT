import json

import pandas as pd
from easyocr.DBNet.DBNet import cv2

impossible_list = ["2003", "2007", "2077", "2018", "2055", "2065", "2103", "2134", "2144", "2149", "2166", "2171", "2172", "2177"]

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
        return found_index
    curr_index = len(arr) // 2
    if arr[curr_index] > timestamp:
        return get_closest_frame(timestamp, arr[0:curr_index], found_index=found_index)
    elif arr[curr_index] < timestamp:
        return get_closest_frame(timestamp, arr[curr_index:-1], found_index=curr_index + found_index)
    else:
        return curr_index + found_index


def extract_video(offical_time, bib_text, found=False):
    start = offical_time - 6
    end = offical_time + 4
    frame_start = get_closest_frame(start)
    frame_end = get_closest_frame(end)
    if frame_start == frame_end:
        return
    cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    writer = cv2.VideoWriter(f"./debug/bib/{bib_text}_{'found' if found else 'not_found'}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    for i in range(frame_start, frame_end):
        ret, frame = cap.read()
        if not ret:
            print("Broke !")
            break
        writer.write(frame)
    writer.release()
    cap.release()


def reconstruct(dict):
    new_dict = {}
    for key in dict.keys():
        curr = dict[key]
        new_bibs = []
        for curr_conf, curr_bib in dict[key]["bibs"]:
            new_conf = curr_conf
            for conf, bib in dict[key]["bibs"]:
                if bib != curr_bib and bib in curr_bib:
                    new_conf += conf
            new_bibs.append((new_conf, curr_bib))
        new_bibs.sort(key=lambda x: x[0], reverse=True)
        new_dict[new_bibs[0][1]] = {
            "time": curr["time"],
            "confidence": new_bibs[0][0],
            "bibs": new_bibs,
        }

    return new_dict


# new_res = reconstruct(results_computed)

# with open("result_reconstructed.json", "w") as file:
#     file.write(json.dumps(new_res))

max_time = max(times) + 60

min_time = min(times)
print(min_time)
print(max_time)
bib_found = 0
bib_not_found = []
count = 0
for key in results_computed.keys():
    if "." in key:
        continue
    if key not in results.keys():
        count += 1

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
        bib_not_found.append(key)
        # extract_videso(results[key], key)
    if key not in impossible_list:
        total += 1


mean_diff /= total_found

print(f"Min diff: {min_diff} - max diff : {max_diff}")
print(f"Found {bib_found} of {total} bibs")
print(f"Mean diff is : {mean_diff}")
print(f"{sorted(bib_not_found)}")
# extract_video(1749888815.291, "2201", found=True)
