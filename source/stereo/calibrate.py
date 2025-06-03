import os
import re
import cv2 as cv
import numpy as np
from typing import TypedDict
import glob
import matplotlib.pyplot as plt

CALIB_PATTERN = (7, 7)
IMG_SIZE = (1080, 1920)

class Calibration(TypedDict):
    mtx: list[list]
    dist: list[list]



def calibrate_camera(folderPath, obj=CALIB_PATTERN, debug=False):
    objp = np.zeros((obj[0] * obj[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:obj[1], 0:obj[0]].T.reshape(-1, 2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 0.0001)
    colors_pics = []

    imgs_corners = []
    obj_points = []
    for filename in os.listdir(folderPath):
        assert filename.endswith(".png")
        filepath = os.path.join(folderPath, filename)
        colors = None
        if debug:
            colors = cv.imread(filepath)
        frame = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        assert frame.shape == IMG_SIZE
        assert frame is not None
        ret, corners = cv.findChessboardCorners(frame, obj)
        assert len(corners) == len(objp)
        if not ret:
            print(f"Chessboard not detected in file : {filepath}")
            continue

        if debug and colors is not None:
            print(f"Frame : {filename}")
            colors_pics.append(colors)
            drawed = cv.drawChessboardCorners(colors, obj, corners, ret)
            cv.imwrite(f"./images/debug/{filename}", drawed)
            # cv.imshow(f"frame", drawed)
            # cv.waitKey(-1)

        corners_refined = cv.cornerSubPix(
            frame,
            corners,
            winSize=(30, 30),
            zeroZone=(-1, -1),
            criteria=criteria
        )
        # corners_refined = corners

        obj_points.append(objp)
        imgs_corners.append(corners_refined)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, imgs_corners, IMG_SIZE, None, None)

    if not ret:
        raise Exception("Calibration failed...")
    else:
        print(f"Calibration precision is : {ret} pixels")

    errors = []
    for i in range(len(obj_points)):
        img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgs_corners[i], img_points2, cv.NORM_L2)/len(img_points2)
        errors.append(error)
        if debug and colors_pics is not None:
            for point in img_points2:
                center = (int(point[0][0]), int(point[0][1]))
                cv.circle(colors_pics[i], center, 5, (0, 255, 0), -1)
            undistorted = cv.undistort(colors_pics[i], mtx, dist)
            cv.imwrite(f"./images/debug/undistorted_{i}.png", undistorted)
            cv.waitKey(-1)
            print(f"Image {i} error: {error:.2f} px")

    return mtx, dist, rvecs, tvecs

def get_depth_maps(folder, Kl, Dl, Kr, Dr, R, T):
    res_folder = "./images/depth_map"
    max_depth = 300
    min_depth = 0.1

    # Create stereo matcher (adjust parameters to your setup)
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Should be divisible by 16
        blockSize=5,
        P1=8*3*5**2,
        P2=32*3*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
	# Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        Kl, Dl,
        Kr, Dr,
        IMG_SIZE, R, T
    )

    # Create rectification maps
    map1x, map1y = cv.initUndistortRectifyMap(
        Kl, Dl, R1, P1, IMG_SIZE, cv.CV_32FC1
    )
    map2x, map2y = cv.initUndistortRectifyMap(
        Kr, Dr, R2, P2, IMG_SIZE, cv.CV_32FC1
    )

    left_pics = list(sorted(glob.glob(os.path.join(folder, "left*.png"))))
    right_pics = list(sorted(glob.glob(os.path.join(folder, "right*.png"))))
    assert len(left_pics) == len(right_pics)

    pic_n = 0

    for left_img_path, right_img_path in zip(left_pics, right_pics):
        left_img = cv.imread(left_img_path, cv.IMREAD_GRAYSCALE)
        right_img = cv.imread(right_img_path, cv.IMREAD_GRAYSCALE)
        assert left_img.shape == IMG_SIZE
        assert right_img.shape == IMG_SIZE
        # Apply rectification
        img1_rect = cv.remap(left_img, map1x, map1y, cv.INTER_LINEAR)
        img2_rect = cv.remap(right_img, map2x, map2y, cv.INTER_LINEAR)

        # Draw horizontal lines to verify rectification
        h, w = img1_rect.shape[:2]
        for y in range(0, h, 50):
            cv.line(img1_rect, (0, y), (w, y), (0, 255, 0), 1)
            cv.line(img2_rect, (0, y), (w, y), (0, 255, 0), 1)

        # Combine side-by-side
        vis_rect = np.hstack((img1_rect, img2_rect))
        cv.imwrite("rectified.png", vis_rect)
        # Compute disparity
        disparity = stereo.compute(img1_rect, img2_rect).astype(np.float32)/16.0
        points_3d = cv.reprojectImageTo3D(disparity, Q)

        depth_map = points_3d[:, :, 2]
        # Create mask of valid depth values
        valid_mask = (disparity > disparity.min()) & (depth_map > min_depth) & (depth_map < max_depth)
        # Clip depth values for visualization
        depth_map[~valid_mask] = 0
        depth_map = np.clip(depth_map, min_depth, max_depth)
        # Normalize for visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(depth_map, cmap='jet', vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))
        plt.colorbar(label='Depth (units)')
        plt.title('Depth Map')
        plt.savefig(res_folder + f"/depth_{pic_n}.png")
        plt.close()
        pic_n += 1

def show_corners(frame, corners):
    cv.imshow("frame", cv.drawChessboardCorners(frame, CALIB_PATTERN, corners, True))
    cv.waitKey()

def stereo_calibration(folder, pattern=CALIB_PATTERN, debug=False):
    left_pics = list(sorted(glob.glob(os.path.join(folder, "left*.png"))))
    right_pics = list(sorted(glob.glob(os.path.join(folder, "right*.png"))))
    assert len(left_pics) == len(right_pics)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    left_pts, right_pts = [], []

    left_image = []
    right_image = []

    for left_img_path, right_img_path in zip(left_pics, right_pics):
        left_img = cv.imread(left_img_path, cv.IMREAD_GRAYSCALE)
        right_img = cv.imread(right_img_path, cv.IMREAD_GRAYSCALE)
        assert left_img.shape == IMG_SIZE
        assert right_img.shape == IMG_SIZE
        left_image = left_img
        right_image = right_img

        res_left, corners_left = cv.findChessboardCorners(left_img, pattern)
        res_right, corners_right = cv.findChessboardCorners(right_img, pattern)

        corners_left = cv.cornerSubPix(left_img, corners_left, (10, 10), (-1,-1), criteria)
        corners_right = cv.cornerSubPix(right_img, corners_right, (10, 10), (-1,-1), criteria)

        # print(f"Image is : {left_img_path}")
        # show_corners(left_img, corners_left)
        # print(f"Image is : {right_img_path}")
        # show_corners(right_img, corners_right)

        left_pts.append(corners_left)
        right_pts.append(corners_right)

    pattern_points = np.zeros((np.prod(pattern), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern).T.reshape(-1, 2)
    pattern_points = [pattern_points] * len(left_pics)

    flags = (
    cv.CALIB_SAME_FOCAL_LENGTH
    )

    err, Kl, Dl, Kr, Dr, R, T, E, F = cv.stereoCalibrate(pattern_points, left_pts, right_pts, None, None, None, None, IMG_SIZE, flags=flags)
    if not err:
        raise Exception("Something went wrong with the stereo calibration...")

    print(f"Stereo calibration error : {err}")
    print(f"Baseline length: {np.linalg.norm(T)} units")
    return Kl, Dl, Kr, Dr, R, T



# mtx_l, dist_l, _, _ = calibrate_camera("./images/left", debug=False)
# mtx_r, dist_r, _, _ = calibrate_camera("./images/right", debug=False)
Kl, Dl, Kr, Dr, R, T = stereo_calibration("./images/stereo")
get_depth_maps("./image/stereo", Kl, Dl, Kr, Dr, R, T)
