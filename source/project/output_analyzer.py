import json
import math

import cv2
import pandas as pd
from dateutil.parser import parse

from classes.tools import get_colored_logger

logger = get_colored_logger(__name__)


class OutputAnalyzer:
    def __init__(self, official_result_filepath: str, computed_results_filepath: str, debug_video_path: str, frame_to_timestamp_filepath: str):
        self.computed_results = self.load_computed_results(computed_results_filepath)
        self.video_cap = cv2.VideoCapture(debug_video_path)
        self.frame_arr = self.load_frames_dict(frame_to_timestamp_filepath)
        self.frame_start = self.computed_results["frame_start"]
        self.frame_end = self.computed_results["frame_end"]
        del self.computed_results["frame_start"]
        del self.computed_results["frame_end"]
        self.FPS = 30
        self.allowed_time_error = 10  # Time is in seconds
        self.compute_timestamp_end = self.frame_arr[self.frame_end]
        self.compute_timestamp_start = self.frame_arr[self.frame_start]
        self.official_results = self.load_official_results(official_result_filepath)

    def load_official_results(self, filepath: str):
        frame = pd.read_csv(filepath, delimiter=";", dtype={"Dossard": "string", "time": "float"})

        dict = {}

        df = pd.DataFrame()

        df["time"] = frame["Heure"].apply(lambda x: parse(x).astimezone(tz=None).timestamp())
        df["bib_n"] = frame["Dossard"]
        dict = {}
        for index, row in df.iterrows():
            if row["time"] < self.compute_timestamp_end and row["time"] > self.compute_timestamp_start:
                dict[row["bib_n"]] = float(row["time"])
        logger.info(f"Loaded official results. {len(dict)} racer have finished during recorded time !")
        return dict

    def load_computed_results(self, filepath: str):
        with open(filepath) as file:
            return json.loads("\n".join(file.readlines()))

    def load_frames_dict(self, filepath):
        with open(filepath) as file:
            lines = file.readlines()
            return [float(l.split(";")[-1]) for l in lines]

    def show_video_clip(self, timestamp, length=10):
        frame = self.get_closest_frame(timestamp, self.frame_arr)
        frame_start = frame - length * self.FPS // 2
        frame_end = math.floor(min(frame_start + length * self.FPS, self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        print(f"Frames : {frame_start + self.frame_start} - {frame_end + self.frame_start}")
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        i = 0
        while i < frame_end:
            ret, frame = self.video_cap.read()
            assert ret
            cv2.imshow("frame", frame)
            key = cv2.waitKey(90)
            if key == ord("q"):
                break
            if key == ord("r"):
                i = frame_start
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            i += 1

    def apply_filtering(self, conf_treshold=1.5):
        to_remove = []
        new_dict = {}
        doublon_count = 0
        removed_conf = 0
        for key, item in self.computed_results.items():
            if item["best_bib"] != "" and item["confidence"] < conf_treshold:
                # to_remove.append(key)
                removed_conf += 1
                continue
            doublon = False
            for key2, item2 in self.computed_results.items():
                if key2 == key:
                    continue
                if item["best_bib"] == item2["best_bib"]:
                    if item["confidence"] < item2["confidence"]:
                        doublon = True
                        new_dict[key] = item
                        new_dict[key]["best_bib"] = ""
                        doublon_count += 1
                        # We already found a bib with a bigger confidence, we can break the loop
                        break
                    # We have a double detection !
            if not doublon:
                new_dict[key] = item
        print(f"Filtering done. Doublon : {doublon_count}, removed with treshold : {removed_conf}")
        self.computed_results = new_dict

    def get_closest_frame(self, timestamp, arr, found_index=0):
        if len(arr) == 1 or len(arr) == 0:
            return found_index - self.frame_start
        curr_index = len(arr) // 2
        if arr[curr_index] > timestamp:
            return self.get_closest_frame(timestamp, arr[0:curr_index], found_index=found_index)
        elif arr[curr_index] < timestamp:
            return self.get_closest_frame(timestamp, arr[curr_index:-1], found_index=curr_index + found_index)
        else:
            return curr_index + found_index - self.frame_start

    def get_statistics(self):
        found_bib = 0
        wrong_detection = 0
        mean_diff = 0
        result = {
            "found_bib": 0,
            "wrong_detection": 0,
            "not_found": 0,
            "mean_diff": 0,
            "no_bib": 0,
            "max_diff": -1,
            "min_diff": -1,
            "detail": {
                "founds": [],
                "not_found": [],
                "wrong_detection": [],
                "no_bib": [],
            },
        }
        found = []
        for key, val in self.computed_results.items():
            bib_text = val["best_bib"]
            computed_time = val["time"]
            if bib_text == "":
                # This mean we have detected someone passing the line, without reading his bib number...
                result["no_bib"] += 1
                result["detail"]["no_bib"].append((None, computed_time, None, key))
            elif bib_text in self.official_results:
                official_time = self.official_results[bib_text]
                time_diff = abs(computed_time - official_time)
                if time_diff > self.allowed_time_error:
                    # Time diff is too big, we didn't read the correct number !
                    result["wrong_detection"] += 1
                    result["detail"]["wrong_detection"].append((bib_text, computed_time, official_time, key))
                else:
                    result["found_bib"] += 1
                    result["mean_diff"] += time_diff
                    result["detail"]["founds"].append((bib_text, computed_time, official_time, key))
                    if result["min_diff"] == -1 or result["min_diff"] > time_diff:
                        result["min_diff"] = time_diff
                    if result["max_diff"] == -1 or result["max_diff"] < time_diff:
                        result["max_diff"] = time_diff
                found.append(bib_text)
            else:
                # We read a number that was not present in the result list...
                result["detail"]["wrong_detection"].append((bib_text, computed_time, None, key))
                result["wrong_detection"] += 1
        for key, value in self.official_results.items():
            if key not in found:
                # We loop to find numbers that we missed
                result["not_found"] += 1
                result["detail"]["not_found"].append((key, None, value))

        # We compute the mean for the bib we found
        if result["found_bib"] != 0:
            result["mean_diff"] /= result["found_bib"]
        return result


def compare_results_details(res1, res2):
    for key in ["founds", "wrong_detection", "not_found"]:
        curr_res1 = [el[0] for el in res1["detail"][key]]
        curr_res2 = [el[0] for el in res2["detail"][key]]
        res1_bonus = []
        res2_bonus = []
        for bib in curr_res1:
            if bib not in curr_res2:
                res1_bonus.append(bib)
        for bib in curr_res2:
            if bib not in curr_res1:
                res2_bonus.append(bib)
        print(f"For {key}:")
        print(f"Present in 1 but not 2 : {res1_bonus}")
        print(f"Present in 2 but not 1 : {res2_bonus}")


parser = OutputAnalyzer(
    official_result_filepath="./data/race_results/official_base.csv",
    computed_results_filepath="./results/results_disco_2/results.json",
    debug_video_path="./results/results_disco_2/output.mp4",
    frame_to_timestamp_filepath="./data/recorded/merged/right_merged_full.csv",
)

res = parser.get_statistics()

parser.apply_filtering()

res2 = parser.get_statistics()

# compare_results_details(res, res2)


for key in ["found_bib", "wrong_detection", "not_found", "no_bib"]:
    print(f"Not filtered : {key} : {res[key]}")
    print(f"Filtered : {key} : {res2[key]}")


def wrong_detection_analysis(res):
    for i in res["detail"]["wrong_detection"]:
        if i[2] is None:
            print(f"Detected a number that does not exists : {i[0]} was person id : {i[3]}")
            parser.show_video_clip(timestamp=i[1])
        else:
            print(f"Detected {i[0]} here. Person id : {i[3]}")
            parser.show_video_clip(timestamp=i[1])
            print("But it is here")
            parser.show_video_clip(timestamp=i[2])


def not_found_analysis(res):
    for i in res["detail"]["not_found"]:
        print(f"{i[0]} not found")
        parser.show_video_clip(timestamp=i[2])


def no_bib_analysis(res):
    for i in res["detail"]["no_bib"]:
        print(f"{i[3]} passed line without bib")
        parser.show_video_clip(timestamp=i[1])


# wrong_detection_analysis(res2)

no_bib_analysis(res2)
