import cv2 as cv
import numpy as np
from camera import depth_estimation

def set_cap_property(cap):
    # Disable auto-exposure (0 = manual, 1 = auto)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    # Disable autofocus (0 = manual, 1 = auto)
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_GAIN, 0)

    cap.set(cv.CAP_PROP_AUTO_WB, 0)
    cap.set(cv.CAP_PROP_WB_TEMPERATURE, 4000)

    # Try different exposure values (varies by camera)
    cap.set(cv.CAP_PROP_FOCUS, 0)  # 50 = mid-range focus
    pass

cap_left = cv.VideoCapture(0)
cap_right = cv.VideoCapture(2)



set_cap_property(cap_left)
set_cap_property(cap_right)

cap_left.set(cv.CAP_PROP_EXPOSURE, 250)  # Example: -4 = moderate exposure
cap_right.set(cv.CAP_PROP_EXPOSURE, 250)  # Example: -4 = moderate exposure

ret_left, frame_left = cap_left.read()
ret_right, frame_right = cap_right.read()

calib = np.load("calib.npz")

left_map1, left_map2, right_map1, right_map2, Q = calib["left_map1"], calib["left_map2"], calib["right_map1"], calib["right_map2"], calib["Q"]

depth_estimation(frame_left, left_map1, left_map2, frame_right, right_map1, right_map2, Q)
