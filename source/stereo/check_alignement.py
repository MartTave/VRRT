import glob
import os
from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from typing_extensions import Sequence

from .calibration import get_charuco_board, get_charuco_detector, get_common_corners, get_points

CALIBRATION_FILE = "./calibrations/calib_1920x1080.npz"

CHECK_FOLDER = "./pic_check/"

file = np.load(CALIBRATION_FILE)

K1, D1, K2, D2, R, T = file["K1"], file["D1"], file["K2"], file["D2"], file["R"], file["T"]


def compute_stereo_reprojection_error(object_points, image_points_left, image_points_right):
    # Project 3D points to left and right images
    left_projected, _ = cv2.projectPoints(object_points, np.eye(3), np.zeros(3), K1, D1)
    right_projected, _ = cv2.projectPoints(object_points, R, T, K2, D2)

    # Calculate Euclidean distances
    left_error = np.linalg.norm(image_points_left - left_projected.squeeze(), axis=1)
    right_error = np.linalg.norm(image_points_right - right_projected.squeeze(), axis=1)

    # Return RMSE
    return np.sqrt(np.mean(np.concatenate([left_error**2, right_error**2])))


left_pics = list(sorted(glob.glob(os.path.join(CHECK_FOLDER, "left*.png"))))
right_pics = list(sorted(glob.glob(os.path.join(CHECK_FOLDER, "right*.png"))))

board = get_charuco_board()

charuco_detector = get_charuco_detector()

for i, (left, right) in enumerate(zip(left_pics, right_pics)):
    # Take new stereo images
    left_img = cv2.imread(left)
    right_img = cv2.imread(right)

    left_corners, left_ids, left_obj, left_img = get_points(left_img, board, charuco_detector)
    right_corners, right_ids, right_obj, right_img = get_points(right_img, board, charuco_detector)

    common_corners_left, common_corners_right, common_ids = get_common_corners(left_corners, left_ids, right_corners, right_ids)

    assert len(common_corners_right) == len(common_corners_left) and len(common_corners_right) == len(common_ids)

    if len(common_corners_right) < 4:
        print("Removed pic, not enough points for stereo alignement check")
        continue

    left_obj_points, left_img_points = board.matchImagePoints(cast(Sequence[MatLike], common_corners_left), common_ids)

    right_obj_points, right_img_points = board.matchImagePoints(cast(Sequence[MatLike], common_corners_right), common_ids)

    assert np.array_equal(left_obj_points, right_obj_points)

    error = compute_stereo_reprojection_error(left_obj_points, left_img_points, right_img_points)
    print(f"Error is : {error}")
