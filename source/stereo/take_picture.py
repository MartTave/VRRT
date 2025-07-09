import os

import cv2 as cv

LEFT_ID = 2
RIGHT_ID = 4
IMAGE_FOLDER = "./data/pics"

LEFT_FOLDER = os.path.join(IMAGE_FOLDER, "left")
RIGHT_FOLDER = os.path.join(IMAGE_FOLDER, "right")
STEREO_FOLDER = os.path.join(IMAGE_FOLDER, "stereo")


def create_folders():
    if not os.path.exists(LEFT_FOLDER):
        os.mkdir(LEFT_FOLDER)
        pass
    if not os.path.exists(RIGHT_FOLDER):
        os.mkdir(RIGHT_FOLDER)
        pass
    if not os.path.exists(STEREO_FOLDER):
        os.mkdir(STEREO_FOLDER)
        pass


def set_cap_property(cap):
    # Disable autofocus (0 = manual, 1 = auto)
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv.CAP_PROP_FOCUS, 0)  # 50 = mid-range focus
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    pass


create_folders()

cap_left = cv.VideoCapture(LEFT_ID, cv.CAP_V4L2)
cap_right = cv.VideoCapture(RIGHT_ID, cv.CAP_V4L2)

set_cap_property(cap_left)
set_cap_property(cap_right)


def build_filename(filename, index):
    splitted = filename.split(".")
    return ".".join(splitted[:-1]) + f"_{index}." + splitted[-1]


def update_index(filepath, curr_index=0):
    full_filepath = build_filename(filepath, curr_index)

    if not os.path.exists(full_filepath):
        return curr_index
    else:
        return update_index(filepath, curr_index + 1)


leftIndex = update_index(os.path.join(LEFT_FOLDER, "picture.png"))
rightIndex = update_index(os.path.join(RIGHT_FOLDER, "picture.png"))
stereoIndex = update_index(os.path.join(STEREO_FOLDER, "left.png"))

index = 0


while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    tookPic = False
    if not ret_left or not ret_right:
        raise Exception(f"Something went wrong during frame capture. Left : {ret_left} right: {ret_right}")

    # Combine the frames horizontally
    resized_left = cv.resize(frame_left, (768, 432))
    resized_right = cv.resize(frame_right, (768, 432))
    combined = cv.hconcat([resized_left, resized_right])

    # Display the combined frame
    cv.imshow("Calib", combined)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("a"):
        cv.imwrite(build_filename(os.path.join(LEFT_FOLDER, "picture.png"), leftIndex), frame_left)
        leftIndex += 1

        cv.imwrite(build_filename(os.path.join(RIGHT_FOLDER, "picture.png"), rightIndex), frame_right)
        rightIndex += 1

        cv.imwrite(build_filename(os.path.join(STEREO_FOLDER, "left.png"), stereoIndex), frame_left)
        cv.imwrite(build_filename(os.path.join(STEREO_FOLDER, "right.png"), stereoIndex), frame_right)
        stereoIndex += 1
        tookPic = True
    elif key == ord("l"):
        cv.imwrite(build_filename(os.path.join(LEFT_FOLDER, "picture.png"), leftIndex), frame_left)
        leftIndex += 1
        tookPic = True
    elif key == ord("r"):
        cv.imwrite(build_filename(os.path.join(RIGHT_FOLDER, "picture.png"), rightIndex), frame_right)
        rightIndex += 1
        tookPic = True
    elif key == ord("s"):
        cv.imwrite(build_filename(os.path.join(STEREO_FOLDER, "left.png"), stereoIndex), frame_left)
        cv.imwrite(build_filename(os.path.join(STEREO_FOLDER, "right.png"), stereoIndex), frame_right)
        stereoIndex += 1
        tookPic = True
    if tookPic:
        print(f"Took pic : {index}")
        index += 1
