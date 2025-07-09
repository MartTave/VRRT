import os

import cv2
import numpy as np

CALIBRATION_FOLDER = "./calibrations"

PIC_SIZE = (1920, 1080)


def get_depth_map(filename, img_left, img_right):
    file = np.load(os.path.join(CALIBRATION_FOLDER, filename))

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    map_left_x, map_left_y = (file["map_left_x"], file["map_left_y"])

    Q = file["Q"]
    f = file["left_mtx"][0, 0]
    B = np.linalg.norm(file["T"])

    print("Q matrix:\n", Q)

    map_right_x, map_right_y = (file["map_right_x"], file["map_right_y"])

    stereo = cv2.StereoBM.create(numDisparities=128, blockSize=21)

    # stereo = cv2.StereoSGBM.create(
    #     minDisparity=0,
    #     numDisparities=40 * 16,  # Must be divisible by 16
    #     blockSize=5,
    #     P1=8 * 3 * 5**2,
    #     P2=32 * 3 * 5**2,
    #     disp12MaxDiff=30,
    #     uniquenessRatio=15,
    #     speckleWindowSize=12,
    #     speckleRange=6,
    # )

    rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)

    rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    resized_left = cv2.resize(rect_img_left, PIC_SIZE)
    resized_right = cv2.resize(rect_img_right, PIC_SIZE)

    concat = cv2.hconcat([resized_left, resized_right])
    for i in range(0, PIC_SIZE[1], 10):
        color = (255, 0, 0)
        if i % 20 == 0:
            color = (0, 0, 255)
        cv2.line(concat, (0, i), (PIC_SIZE[0] * 2, i), color, 1)

    # cv2.imshow("Concat", concat)
    cv2.imwrite("concat.jpg", concat)
    # while True:
    #     key = cv2.waitKey(-1)
    #     if key == ord("q"):
    #         break

    disparity = stereo.compute(rect_img_left, rect_img_right).astype(np.float32) / 16.0

    valid_mask = disparity > 0  # Only keep positive disparities
    filtered_disparity = np.zeros_like(disparity)
    filtered_disparity[valid_mask] = disparity[valid_mask]

    # filtered_disp = wls_filter.filter(disparity, rect_img_left, None, disparity_right)

    print("Disparity min/max:", disparity.min(), disparity.max())

    # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Extract depth map (Z coordinate)
    curr_depth = points_3D[:, :, 2]
    curr_depth = points_3D[:, :, 2]
    min_depth = 0.1  # 10cm minimum (adjust to your scene)
    max_depth = 10000  # 5m maximum (adjust to your scene)
    curr_depth[curr_depth < min_depth] = min_depth
    curr_depth[curr_depth > max_depth] = max_depth
    curr_depth[~valid_mask] = max_depth

    curr_depth[np.isinf(curr_depth)] = 10000

    print("Depth min/max:", curr_depth.min(), curr_depth.max())

    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    # Normalize for display
    depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

    # resized = cv2.resize(disparity_vis, (1280, 720))

    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    # cv2.imshow("Disparity", disparity_color)
    # cv2.imshow("Depth", depth_color)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("disp", 800, 1200)

    def nothing(x):
        pass

    cv2.createTrackbar("numDisparities", "disp", 40, 150, nothing)
    cv2.createTrackbar("blockSize", "disp", 1, 100, nothing)
    cv2.createTrackbar("preFilterType", "disp", 0, 1, nothing)
    cv2.createTrackbar("preFilterSize", "disp", 1, 100, nothing)
    cv2.createTrackbar("preFilterCap", "disp", 1, 62, nothing)
    cv2.createTrackbar("textureThreshold", "disp", 7, 100, nothing)
    cv2.createTrackbar("uniquenessRatio", "disp", 1, 100, nothing)
    cv2.createTrackbar("speckleRange", "disp", 6, 150, nothing)
    cv2.createTrackbar("speckleWindowSize", "disp", 6, 75, nothing)
    cv2.createTrackbar("disp12MaxDiff", "disp", 30, 250, nothing)
    cv2.createTrackbar("minDisparity", "disp", 200, 1000, nothing)

    while True:
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos("numDisparities", "disp") * 16 + 16
        blockSize = cv2.getTrackbarPos("blockSize", "disp") * 2 + 5
        preFilterType = cv2.getTrackbarPos("preFilterType", "disp")
        preFilterSize = cv2.getTrackbarPos("preFilterSize", "disp") * 2 + 5
        preFilterCap = cv2.getTrackbarPos("preFilterCap", "disp") + 1
        textureThreshold = cv2.getTrackbarPos("textureThreshold", "disp")
        uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disp")
        speckleRange = cv2.getTrackbarPos("speckleRange", "disp")
        speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disp") * 2
        disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disp")
        minDisparity = cv2.getTrackbarPos("minDisparity", "disp")

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(rect_img_left, rect_img_right)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
        # points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # # Extract depth map (Z coordinate)
        # curr_depth = points_3D[:, :, 2]

        # depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # cv2.imshow("depth", depth_vis)
        # Displaying the disparity map
        cv2.imshow("disp", disparity)
        if cv2.waitKey(10) == ord("q"):
            break


def set_cap_property(cap):
    # Disable autofocus (0 = manual, 1 = auto)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FOCUS, 0)  # 50 = mid-range focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    pass


def depth_map(filename):
    file = np.load(os.path.join(CALIBRATION_FOLDER, filename))

    map_left_x, map_left_y = (file["map_left_x"], file["map_left_y"])

    Q = file["Q"]

    map_right_x, map_right_y = (file["map_right_x"], file["map_right_y"])

    cap_left = cv2.VideoCapture(2)

    cap_right = cv2.VideoCapture(4)

    cv2.namedWindow("Disparity Map")

    nothing = lambda x: None

    cv2.createTrackbar("NumDisparities", "Disparity Map", 1, 50, nothing)
    cv2.createTrackbar("BlockSize", "Disparity Map", 5, 50, nothing)

    set_cap_property(cap_left)
    set_cap_property(cap_right)

    ret_left, frame = cap_left.read()

    ret_right, frame = cap_right.read()

    assert ret_left
    assert ret_right

    stereo_left = cv2.StereoBM.create(numDisparities=32, blockSize=7)
    filter = cv2.ximgproc.DisparityWLSFilter(matcher_left=stereo_left)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo_left)

    while True:
        ret, img_left = cap_left.read()
        ret, img_right = cap_right.read()
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)

        rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        disparity_left = stereo_left.compute(rect_img_left, rect_img_right)
        disparity_right = right_matcher.compute(rect_img_right, rect_img_left)

        # Q is the 4x4 disparity-to-depth mapping matrix from stereoRectify
        points_3D = cv2.reprojectImageTo3D(disparity_left, Q)

        # Extract depth map (Z coordinate)
        curr_depth = points_3D[:, :, 2]

        disparity_left = filter.filter(disparity_left, img_left, None, disparity_right)

        disparity_vis = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

        # Normalize for display
        depth_vis = cv2.normalize(curr_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore

        resized = cv2.resize(disparity_vis, PIC_SIZE)

        cv2.imshow("Disparity Map", resized)
        cv2.waitKey(1)

        pass


img1 = cv2.imread("./data/pics/stereo/left_30.png")
img2 = cv2.imread("./data/pics/stereo/right_30.png")

get_depth_map("calib_1.npz", img1, img2)
