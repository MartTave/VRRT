import json

import cv2

from classes.depth import get_arrival_line

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


FRAME = 135 * 60 * 30
cap = cv2.VideoCapture("./data/recorded/merged/right_merged.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ret, frame = cap.read()
assert ret

parameters = {}

crop_points = get_cropping_region(frame)
if crop_points is None:
    raise Exception("You need to select a cropping region")

parameters["crop"] = [crop_points[0], crop_points[1]]
frame = frame[crop_points[0][1] : crop_points[1][1], crop_points[0][0] : crop_points[1][0]]

print("Please click to designate an arrival line, and thèen press 'S' to save")
line = get_arrival_line(frame)

parameters["line"] = {
    "start": line[0],
    "end": line[1],
}


with open("parameters.json", "w") as file:
    file.write(json.dumps(parameters))
