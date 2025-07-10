import json
import os

import cv2
import numpy as np
from classes.depth import get_arrival_line

# Those are the parameters to change for this script
FOLDER = "./parameters"  # The output folder for the parameters file
FRAME = 0  # The frame of the video to do the parameters description to

# Select one of the two sources below
VIDEO_FILE = "./data/depth_precision/depth_benchmark_1.mp4"  # The path to the video file
PICTURE_FILE = ""
CAMERA_INDEX = -1

points = []
clone = []


def get_cropping_region(frame):
    global points, clone
    # Global variables
    clone = frame.copy()
    window_name = "image"
    points = []

    def click_and_crop(event, x, y, flags, param):
        global points, clone
        # If the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            # If we have more than two points, reset
            if len(points) > 2:
                clone = frame.copy()
                points = [(x, y)]
                cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow(window_name, clone)

            # Draw the point
            cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, clone)

            # If we have two points, draw the rectangle
            if len(points) == 2:
                cv2.rectangle(clone, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow(window_name, clone)

    # Create a window and set the mouse callback function
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_and_crop)

    # Keep looping until 'q' is pressed or 's' to save
    while True:
        # Display the image and wait for a keypress
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF

        # If 's' is pressed and we have exactly two points, return the coordinates
        if key == ord("s"):
            if len(points) == 2:
                cv2.destroyAllWindows()
                # Ensure points are in (top-left, bottom-right) order
                x1, y1 = points[0]
                x2, y2 = points[1]
                top_left = (min(x1, x2), min(y1, y2))
                bottom_right = (max(x1, x2), max(y1, y2))
                return (top_left, bottom_right)
            else:
                print("You need to select exactly two points before saving!")

        # If 'q' is pressed, exit without saving
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None


frame = np.array([])
parameters = {}


if VIDEO_FILE != "":
    cap = cv2.VideoCapture(VIDEO_FILE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ret, curr_frame = cap.read()
    assert ret
    frame = curr_frame

if PICTURE_FILE != "":
    frame = cv2.imread(PICTURE_FILE)

if CAMERA_INDEX != -1:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, curr_frame = cap.read()
    assert ret
    frame = curr_frame

crop_points = get_cropping_region(frame)
if crop_points is None:
    raise Exception("You need to select a cropping region")

parameters["crop"] = [crop_points[0], crop_points[1]]
frame = frame[crop_points[0][1] : crop_points[1][1], crop_points[0][0] : crop_points[1][0]]

print("Please click to designate an arrival line, and then press 'S' to save")
line = get_arrival_line(frame)

parameters["line"] = {
    "start": line[0],
    "end": line[1],
}

index = 0
curr_path = ""
while True:
    curr_path = os.path.join(FOLDER, f"parameters_{index}.json")
    if not os.path.exists(curr_path):
        break
    index += 1


with open(curr_path, "w") as file:
    file.write(json.dumps(parameters))

print(f"Parameters saved to :  {curr_path}")
