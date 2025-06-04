import os

import cv2
import numpy as np

CALIBRATION_FOLDER = "./calibrations"


def get_depth_map(filename, img_left, img_right):
    file = np.load(os.path.join(CALIBRATION_FOLDER, filename))

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    map_left_x, map_left_y = (file["map_left_x"], file["map_left_y"])

    Q = file["Q"]

    print("Q matrix:\n", Q)

    map_right_x, map_right_y = (file["map_right_x"], file["map_right_y"])

    stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)

    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=64,
        blockSize=3,
        P1=8 * 3 * 3,
        P2=32 * 3 * 3,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)

    rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    # resized_left = cv2.resize(rect_img_left, (640, 480))
    # resized_right = cv2.resize(rect_img_right, (640, 480))

    # concat = cv2.hconcat([resized_left, resized_right])
    # for i in range(0, 480, 10):
    #     color = (255, 0, 0)
    #     if i % 20 == 0:
    #         color = (0, 0, 255)
    #     cv2.line(concat, (0, i), (1280, i), color, 1)

    # cv2.imshow("Concat", concat)
    # while True:
    #     key = cv2.waitKey(-1)
    #     if key == ord("q"):
    #         break

    disparity = stereo.compute(rect_img_left, rect_img_right)

    print("Disparity min/max:", disparity.min(), disparity.max())

    # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Extract depth map (Z coordinate)
    curr_depth = points_3D[:, :, 2]

    curr_depth[np.isinf(curr_depth)] = 20000

    print("Depth min/max:", curr_depth.min(), curr_depth.max())

    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    # Normalize for display
    depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    # resized = cv2.resize(disparity_vis, (1280, 720))

    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("Disparity", disparity_color)
    cv2.imshow("Depth", depth_color)

    while True:
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break
    pass


def set_cap_property(cap):
    # Disable autofocus (0 = manual, 1 = auto)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FOCUS, 0)  # 50 = mid-range focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)

    rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    resized_left = cv2.resize(rect_img_left, (640, 480))
    resized_right = cv2.resize(rect_img_right, (640, 480))

    concat = cv2.hconcat([resized_left, resized_right])
    for i in range(0, 480, 10):
        color = (255, 0, 0)
        if i % 20 == 0:
            color = (0, 0, 255)
        cv2.line(concat, (0, i), (1280, i), color, 1)

    cv2.imshow("Concat", concat)
    while True:
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break

    disparity = stereo.compute(rect_img_left, rect_img_right)

    # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Extract depth map (Z coordinate)
    curr_depth = points_3D[:, :, 2]

    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    # Normalize for display
    depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    resized = cv2.resize(disparity_vis, (1280, 720))

    cv2.imshow("disparity", rect_img_left)
    cv2.waitKey(1)
    pass


def depth_map(filename):
    file = np.load(os.path.join(CALIBRATION_FOLDER, filename))

    map_left_x, map_left_y = (file["map_left_x"], file["map_left_y"])

    Q = file["Q"]

    map_right_x, map_right_y = (file["map_right_x"], file["map_right_y"])

    cap_left = cv2.VideoCapture(2)

    cap_right = cv2.VideoCapture(4)

    set_cap_property(cap_left)
    set_cap_property(cap_right)

    ret_left, frame = cap_left.read()

    ret_right, frame = cap_right.read()

    assert ret_left
    assert ret_right

    stereo = cv2.StereoBM.create(numDisparities=16, blockSize=128)

    while True:
        ret, img_left = cap_left.read()

        ret, img_right = cap_right.read()

        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)

        rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        resized_left = cv2.resize(rect_img_left, (640, 480))
        resized_right = cv2.resize(rect_img_right, (640, 480))

        concat = cv2.hconcat([resized_left, resized_right])
        for i in range(0, 480, 10):
            color = (255, 0, 0)
            if i % 20 == 0:
                color = (0, 0, 255)
            cv2.line(concat, (0, i), (1280, i), color, 1)

        cv2.imshow("Concat", concat)
        while True:
            key = cv2.waitKey(-1)
            if key == ord("q"):
                break

        disparity = stereo.compute(rect_img_left, rect_img_right)

        # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # Extract depth map (Z coordinate)
        curr_depth = points_3D[:, :, 2]

        disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

        # Normalize for display
        depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

        resized = cv2.resize(disparity_vis, (1280, 720))

        cv2.imshow("disparity", rect_img_left)
        cv2.waitKey(1)

        pass


img1 = cv2.imread("./data/pics/stereo/left_0.png")
img2 = cv2.imread("./data/pics/stereo/right_0.png")

get_depth_map("calib_2.npz", img1, img2)
