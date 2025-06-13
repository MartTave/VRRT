import glob
import os
from typing import cast

import cv2
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike
from typing_extensions import Sequence


def get_common_corners(corners1, ids1, corners2, ids2):
    """Return corners and ids present in both images.

    Args:
        corners1: List of corners from first image
        ids1: List of corresponding ids from first image
        corners2: List of corners from second image
        ids2: List of corresponding ids from second image

    Returns:
        common_corners1: Corners from first image that exist in both
        common_corners2: Corners from second image that exist in both
        common_ids: IDs that exist in both images

    """
    assert ids1 is not None and ids2 is not None and len(ids1) != 0 and len(ids2) != 0

    # Convert ids to 1D arrays for easier handling
    ids1_flat = ids1.flatten()
    ids2_flat = ids2.flatten()

    # Find intersection of ids
    common_ids = np.intersect1d(ids1_flat, ids2_flat)

    if len(common_ids) == 0:
        return [], [], []

    # Get indices of common ids in each image
    idx1 = np.where(np.isin(ids1_flat, common_ids))[0]
    idx2 = np.where(np.isin(ids2_flat, common_ids))[0]

    # Ensure the order matches
    # We need to reorder one set of corners so the ids align
    order1 = np.argsort(ids1_flat[idx1])
    order2 = np.argsort(ids2_flat[idx2])

    common_corners1 = np.array(corners1)[idx1][order1]
    common_corners2 = np.array(corners2)[idx2][order2]
    common_ids_sorted = np.sort(common_ids)

    return common_corners1, common_corners2, common_ids_sorted


def get_points(img, board, charuco_detector):
    charuco_corners, charuco_ids, markers_corners, markers_ids = charuco_detector.detectBoard(img)

    charuco_corners = cast(Sequence[MatLike], np.squeeze(charuco_corners))

    charuco_ids = cast(MatLike, charuco_ids)

    try:
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    except:
        print("No arcuo detected in this pics !")
        # cv2.imshow("frame", img)
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == ord("q"):
        #         break
        return None, None, None, None
        # raise Exception("uh oh")

    return charuco_corners, cast(MatLike, charuco_ids), obj_points, img_points


def get_charuco_dict():
    return aruco.getPredefinedDictionary(aruco.DICT_4X4_100)


def get_charuco_board():
    board = aruco.CharucoBoard((7, 5), 37, 27, get_charuco_dict())

    # This is needed as I generated the board with this : https://calib.io/pages/camera-calibration-pattern-generator
    board.setLegacyPattern(True)
    return board


def get_aruco_detector():
    return aruco.ArucoDetector(get_charuco_dict())


def get_charuco_detector():
    return aruco.CharucoDetector(get_charuco_board())


def calibrate_camera(pattern, min_detect=4):
    dictionary = get_charuco_dict()
    board = get_charuco_board()

    detector = get_aruco_detector()

    charuco_detector = get_charuco_detector()

    all_corners = []
    all_ids = []
    for pic in pattern:
        im = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)

        corners, ids, _ = detector.detectMarkers(im)

        if ids is not None and len(ids) > min_detect:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=im, board=board)


def calibrate_from_pics(path, save_results=True):
    left_pics = list(sorted(glob.glob(os.path.join(path, "left*.png"))))
    right_pics = list(sorted(glob.glob(os.path.join(path, "right*.png"))))
    assert len(left_pics) == len(right_pics)
    charuco_detector = get_charuco_detector()

    board = get_charuco_board()

    all_obj_points = {"left": [], "right": []}

    all_img_points = {"left": [], "right": []}

    stereo_points = {"obj_points": [], "left_imgs_points": [], "right_imgs_points": []}

    img_size = None

    for i, (left, right) in enumerate(zip(left_pics, right_pics)):
        left_img = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right, cv2.IMREAD_GRAYSCALE)

        size = (1920, 1080)

        left_img = cv2.resize(left_img, size)
        right_img = cv2.resize(right_img, size)

        if img_size is None:
            img_size = left_img.shape[:2]
            print(f"Image size is : {img_size[0]}x{img_size[1]}")
        assert np.array_equal(img_size, left_img.shape[:2]) and np.array_equal(img_size, right_img.shape[:2])

        left_corners, left_ids, left_obj, left_img = get_points(left_img, board, charuco_detector)
        right_corners, right_ids, right_obj, right_img = get_points(right_img, board, charuco_detector)

        if left_corners is None or right_corners is None:
            continue

        if len(left_corners) > 4:
            all_obj_points["left"].append(left_obj)
            all_img_points["left"].append(left_img)
        if len(right_corners) > 4:
            all_obj_points["right"].append(right_obj)
            all_img_points["right"].append(right_img)

        # The ids list are sorted !
        common_corners_left, common_corners_right, common_ids = get_common_corners(left_corners, left_ids, right_corners, right_ids)

        assert len(common_corners_right) == len(common_corners_left) and len(common_corners_right) == len(common_ids)

        if len(common_corners_right) < 6:
            print("Removed pic, not enough points for stereo calibration")
            continue

        left_obj_points, left_img_points = board.matchImagePoints(cast(Sequence[MatLike], common_corners_left), common_ids)

        right_obj_points, right_img_points = board.matchImagePoints(cast(Sequence[MatLike], common_corners_right), common_ids)

        assert np.array_equal(left_obj_points, right_obj_points)

        stereo_points["obj_points"].append(left_obj_points)
        stereo_points["right_imgs_points"].append(right_img_points)
        stereo_points["left_imgs_points"].append(left_img_points)
        print(f"Pic {i + 1}/{len(left_pics)}")

    print("Processed all pictures, starting calibration")

    assert img_size is not None

    left_ret, left_mtx, left_dist, _, _ = cv2.calibrateCamera(
        all_obj_points["left"], all_img_points["left"], imageSize=img_size, cameraMatrix=None, distCoeffs=None
    )  # type: ignore

    right_ret, right_mtx, right_dist, _, _ = cv2.calibrateCamera(
        all_obj_points["right"], all_img_points["right"], imageSize=img_size, cameraMatrix=None, distCoeffs=None
    )  # type: ignore

    print(f"Camera calibration results : \n Left : {left_ret}\n Right : {right_ret}")

    stereo_ret, new_left_mtx, new_left_dist, new_right_mtx, new_right_dist, R, T, E, F = cv2.stereoCalibrate(
        stereo_points["obj_points"],
        stereo_points["left_imgs_points"],
        stereo_points["right_imgs_points"],
        left_mtx,
        left_dist,
        right_mtx,
        right_dist,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    assert (
        np.array_equal(new_left_mtx, left_mtx)
        and np.array_equal(new_left_dist, left_dist)
        and np.array_equal(new_right_mtx, right_mtx)
        and np.array_equal(new_right_dist, right_dist)
    )

    print(f"Stereo camera calibration results : {stereo_ret}")

    real_size = (img_size[1], img_size[0])
    print(f"Real size is : {real_size}")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=left_mtx,
        distCoeffs1=left_dist,
        cameraMatrix2=right_mtx,
        distCoeffs2=right_dist,
        imageSize=real_size,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(left_mtx, left_dist, R1, P1, real_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(right_mtx, right_dist, R2, P2, real_size, cv2.CV_32FC1)

    if save_results:
        CALIB_FOLDER = "./calibrations"

        curr_files = list(sorted(glob.glob(os.path.join(CALIB_FOLDER, "calib_*.npz"))))

        curr_index = 0

        if len(curr_files) > 0:
            curr_index = int(curr_files[-1][-5]) + 1

        calib_path = f"./calibrations/calib_{curr_index}.npz"

        np.savez_compressed(
            calib_path,
            left_mtx=left_mtx,
            right_mtx=right_mtx,
            left_dist=left_dist,
            right_dist=right_dist,
            R=R,
            T=T,
            E=E,
            F=F,
            Q=Q,
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            map_left_x=map_left_x,
            map_left_y=map_left_y,
            map_right_x=map_right_x,
            map_right_y=map_right_y,
        )
        print(f"New calibration saved at : {calib_path}")


if __name__ == "__main__":
    calibrate_from_pics("./data/pics/stereo/")
