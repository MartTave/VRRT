import csv
import glob
import os

import cv2
import numpy as np

LEFT_VIDEOS = "./records/left"
RIGHT_VIDEOS = "./record/right"


OUTPUT_FOLDER = "./records/merged"
STEREO_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "stereo")

CALIBRATIONS = "./calibrations/calib_14.06.npz"

left_videos = list(sorted(glob.glob(os.path.join(LEFT_VIDEOS, "video_*.mp4"))))
right_videos = list(sorted(glob.glob(os.path.join(RIGHT_VIDEOS, "video_*.mp4"))))


# This function is here to merge mutliples video files in one
# You can use it if you DO NOT need stereo vision
# This means: no syncronization, no rectification etc..
# Only a merge of the timestamps and the videos
def merge_monocular_video(video_folder):
    dest_folder = os.path.join(OUTPUT_FOLDER, "monocular")
    left_text = list(sorted(glob.glob(os.path.join(video_folder, "video_*.txt"))))
    left_videos = list(sorted(glob.glob(os.path.join(video_folder, "video_*.mp4"))))
    global_frame_index = 0
    with open(os.path.join(dest_folder, "merged_2.txt", "w")) as outfile:
        for l_txt, l_vid in zip(left_text, left_videos, strict=False):
            cap = cv2.VideoCapture(l_vid)
            print(f"Doing file : {l_vid}")
            with open(l_txt) as file:
                lines = file.readlines()
                for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    curr_line = lines[i].split(";")
                    timestamp = float(curr_line[1])
                    outfile.write(f"{str(global_frame_index)};{timestamp}\n")
                    global_frame_index += 1
    with open(os.path.join(dest_folder, "filelist.txt"), "w") as outfile:
        for l in left_videos:
            outfile.write(f"file {l}\n")
    print("Concat of timestamp done, you can now do : ")
    print(f"ffmpeg -f concat -safe 0 -i {os.path.join(dest_folder, 'filelist.txt')} -c copy {os.path.join(dest_folder, 'merged.mp4')}")
    print("To merge the video file, this is faster as no processing is needed on the frames themselves")
    pass


# This function is used to merge all the small timestamps files into a single one
def build_merged_timstamp():
    left_text = list(sorted(glob.glob(os.path.join(LEFT_VIDEOS, "video_*.txt"))))

    right_text = list(sorted(glob.glob(os.path.join(RIGHT_VIDEOS, "video_*.txt"))))

    def concat_timestamps(files):
        index = 0
        res = []
        for f in files:
            video_filename = f.replace(".txt", ".mp4")
            with open(f) as file:
                lines = file.readlines()
                for l in lines:
                    splitted = l.split(";")
                    res.append(f"{int(splitted[0])};{float(splitted[1])};{video_filename}\n")
                index += len(lines)
            print(f"Done : {f}")
        return res

    with open(os.path.join(STEREO_OUTPUT_FOLDER, "/left_merged.csv"), "w") as f:
        f.writelines(concat_timestamps(left_text))

    with open(os.path.join(STEREO_OUTPUT_FOLDER, "/right_merged.csv"), "w") as f:
        f.writelines(concat_timestamps(right_text))


left_index = 0
right_index = 0


# This function is used to merge and syncronize left and right timestamps files. The result of this can
def create_synced_timestamps():
    with open(os.path.join(STEREO_OUTPUT_FOLDER, "left_merged.csv"), "r") as left_file:
        with open(os.path.join(STEREO_OUTPUT_FOLDER, "right_merged.csv"), "r") as right_file:
            with open(os.path.join(STEREO_OUTPUT_FOLDER, "synced.csv"), "w") as out_file:
                out_file.write("left_frame_n;left_filename;left_timestamp;right_frame_n;right_filename;right_timestamp;timestamp_diff\n")
                left_lines = left_file.readlines()
                right_lines = right_file.readlines()
                max_diff = 1 / 30 / 2
                global left_index
                global right_index

                def sync_timestamp():
                    global left_index
                    global right_index
                    left_timestamp = float(left_lines[left_index].split(";")[1])
                    right_timestamp = float(right_lines[right_index].split(";")[1])
                    diff = abs(left_timestamp - right_timestamp)
                    if diff > max_diff:
                        # We can find a better timestamp
                        if left_timestamp < right_timestamp:
                            left_index += 1
                        else:
                            right_index += 1
                        print("Timestamps not in sync, trying to sync !")
                        return False
                    else:
                        # we consider that "in sync"
                        return True

                while True:
                    print(f"{left_index} - {right_index}")
                    if left_index >= len(left_lines):
                        break
                    if right_index >= len(right_lines):
                        break
                    if sync_timestamp():
                        left_line = left_lines[left_index].replace("\n", "").split(";")
                        right_line = right_lines[right_index].replace("\n", "").split(";")
                        left_frame_n = left_line[0]
                        right_frame_n = right_line[0]
                        left_filename = left_line[2]
                        right_filename = right_line[2]
                        left_timestamp = float(left_line[1])
                        right_timestamp = float(right_line[1])
                        timestamp_diff = abs(left_timestamp - right_timestamp)
                        left_index += 1
                        right_index += 1
                        out_file.write(
                            f"{left_frame_n};{left_filename};{left_timestamp};{right_frame_n};{right_filename};{right_timestamp};{timestamp_diff}\n"
                        )


captures = {"l": [None, None], "r": [None, None]}

curr_frames = {"l": 1, "r": 1}


# This is the function used to output a single video files, with both videos stream syncronized
# This will rectify the stereos frame too.
def merge_videos():
    global captures
    calib_file = np.load(CALIBRATIONS)

    map_left_x, map_left_y = (calib_file["map_left_x"], calib_file["map_left_y"])
    map_right_x, map_right_y = (calib_file["map_right_x"], calib_file["map_right_y"])

    def get_cap(filename, side) -> cv2.VideoCapture:
        global captures
        assert side == "l" or side == "r"
        if captures[side][0] != filename:
            if captures[side][1] is not None:
                captures[side][1].release()
            captures[side][1] = cv2.VideoCapture(filename)
            captures[side][0] = filename
            curr_frames[side] = 1
        return captures[side][1]

    def get_frame(filename, side, frame_n):
        cap = get_cap(filename, side)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        ret, frame = cap.read()
        if not ret:
            # Get total frames in video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Error while trying to read file : {filename} at position {frame_n}. Total frame is : {total_frames}")
            return False
        return frame

    with open(os.path.join(STEREO_OUTPUT_FOLDER, "synced.csv"), "r") as csvfile:
        content = csv.DictReader(csvfile, delimiter=";")
        writter = cv2.VideoWriter(OUTPUT_FOLDER + "/synced.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920 * 2, 1080), isColor=True)
        for i, row in enumerate(content):
            l_frame = get_frame(row["left_filename"], "l", int(row["left_frame_n"]) - 1)
            r_frame = get_frame(row["right_filename"], "r", int(row["right_frame_n"]) - 1)
            if l_frame is False and r_frame is False:
                continue
            rect_img_left = cv2.remap(l_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
            rect_img_right = cv2.remap(r_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)
            stacked = cv2.hconcat((rect_img_left, rect_img_right))
            writter.write(stacked)
            print(f"Done frame {i}")
    writter.release()
    captures["l"][1].release()
    captures["r"][1].release()


# This is a helper function that uses all the subfunctions in the right order
def full_merge_videos():
    build_merged_timstamp()
    create_synced_timestamps()
    merge_videos()


# This is an example of how to use the function to merge monocular videos
merge_monocular_video("./records/right")
