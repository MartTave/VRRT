import json
import math
import os

import cv2
import pandas as pd
from dateutil.parser import parse

from classes.tools import get_colored_logger

logger = get_colored_logger(__name__)


unreadable = [
    "2083",
    "2171",
    "2172",
    "2065",
    "2112",
    "2018",
    "2055",
    "2007",
    "2170",
    "2139",
    "2163",
    "2077",
    "2046",
    "2178",
    "2159",
    "2048",
    "2025",
    "2029",
    "2041",
    "2073",
    "1129",
]


class OutputAnalyzer:
    def __init__(
        self,
        official_result_filepath: str,
        computed_results_filepath: str,
        debug_video_path: str,
        frame_to_timestamp_filepath: str,
        remove_unreadable=True,
    ):
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
        self.remove_unreadable = remove_unreadable
        if not remove_unreadable:
            self.official_results = self.load_official_results(official_result_filepath)
        else:
            self.official_results = {}
            temp = self.load_official_results(official_result_filepath)
            for key in temp:
                if key not in unreadable:
                    self.official_results[key] = temp[key]

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
        if frame_start + self.frame_start == -150:
            import ipdb

            ipdb.set_trace()
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
        result = {
            "correct": 0,
            "wrong_time": 0,
            "not_found": 0,
            "detail": {
                "correct": [],
                "not_found": [],
                "wrong_time": [],
                "no_bib": [],
            },
        }
        for bib, time in self.official_results.items():
            found = False
            for person_id, obj in self.computed_results.items():
                if obj["best_bib"] == bib:
                    found = True
                    # We found the bib in the computed result
                    time_diff = time - obj["time"]
                    if time_diff > 0 and time_diff < self.allowed_time_error:
                        # We found the correct bib at the correct time !
                        result["correct"] += 1
                    else:
                        result["wrong_time"] += 1
                    break
            if not found:
                result["not_found"] += 1

        return result

    def get_metrics(self, res):
        correct = res["correct"]
        wrong_time = res["wrong_time"]
        missed = res["not_found"]
        total = len(self.official_results)
        print(f"Total bibs to detect : {total}")
        print(f"Correct : {correct} ==> {correct / total:.2f}")
        print(f"Wrong time : {wrong_time} ==> {wrong_time / total:.2f}")
        print(f"Bib missed : {missed} ==> {missed / total:.2f}")


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


run_folder = "./results/runs/run_0"

parser = OutputAnalyzer(
    official_result_filepath="./data/race_results/official_base.csv",
    computed_results_filepath=os.path.join(run_folder, "results.json"),
    debug_video_path=os.path.join(run_folder, "out.mp4"),
    frame_to_timestamp_filepath="./data/recorded/merged/right_merged_full.csv",
)

parser.apply_filtering()

res = parser.get_statistics()


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


parser.get_metrics(res)

# compare_results_details(res, res2)

# wrong_dete&ction_analysis(res2)

# wrong_detection_analysis(res2)

# not_found_analysis(res2)
