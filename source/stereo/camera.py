import cv2 as cv
import numpy as np



# Example: Disparity (using SGBM)
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Adjust based on your setup
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
    )

def depth_estimation(left_img, left_map1, left_map2, right_img, right_map1, right_map2, Q):

    # Check if maps contain valid values (non-zero)
    print("Left map1 min/max:", np.min(left_map1), np.max(left_map1))
    print("Left map2 min/max:", np.min(left_map2), np.max(left_map2))
    print("Right map1 min/max:", np.min(right_map1), np.max(right_map1))
    print("Right map2 min/max:", np.min(right_map2), np.max(right_map2))


    # Rectify images
    left_rectified = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
    right_rectified = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

    # Compute disparity
    disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

    cv.imwrite("images/disparity.png", disparity * 255)
    cv.imwrite("images/left.png", left_img)
    cv.imwrite("images/right.png", right_img)

    cv.imwrite("images/left_rectified.png", left_rectified)
    cv.imwrite("images/right_rectified.png", right_rectified)

    # Convert disparity to depth (Q is from stereoRectify)
    depth = cv.reprojectImageTo3D(disparity, Q)

    # Normalize to 0-255 for visualization
    depth_vis = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply a colormap (Jet/Magma for better perception)
    depth_color = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)

    # Display
    cv.imwrite('images/depth_map.png', depth_color)


def calibrateSetup(left_id, right_id, obj=(13, 9), resolution=(640, 480)):

    def detect_chessboard(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, obj, None)
        return ret, corners

    def calibrate_cam(objpoints, imgpoints):
        # Calibrate left camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, resolution, None, None)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(objpoints)

        return ret, mtx, dist, rvecs, tvecs, mean_error

    def calibrate_stereo(objpoints, imgpoints_left, mtx_left, dist_left, imgpoints_right, mtx_right, dist_right):
        # Stereo calibration
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            resolution, flags=cv.CALIB_FIX_INTRINSIC)
        return ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F

    def compute_rectification_map(mtx_left, dist_left, mtx_right, dist_right, R, T):
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            resolution, R, T, alpha=0.9)

        # Create undistortion/rectification maps
        left_map1, left_map2 = cv.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, resolution, cv.CV_16SC2)
        right_map1, right_map2 = cv.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, resolution, cv.CV_16SC2)
        return left_map1, left_map2, right_map1, right_map2, Q

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    frames_left = []
    frames_right = []

    def stereo_calibration(img_left, img_right):

        ret_left, corners_left = detect_chessboard(img_left)

        ret_right, corners_right = detect_chessboard(img_right)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            ret_left, mtx_left, dist_left, _, _, mean_error_left = calibrate_cam(objpoints=objpoints, imgpoints=imgpoints_left)
            ret_right, mtx_right, dist_right, _, _, mean_error_right = calibrate_cam(objpoints=objpoints, imgpoints=imgpoints_right)

            print(f"Mean error is, left:{mean_error_left} right:{mean_error_right}")

            if ret_left and ret_right:
                ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = calibrate_stereo(objpoints=objpoints,imgpoints_left=imgpoints_left, mtx_left=mtx_left, dist_left=dist_left, imgpoints_right=imgpoints_right, mtx_right=mtx_right, dist_right=dist_right)
                if ret:
                    left_map1, left_map2, right_map1, right_map2, Q = compute_rectification_map(mtx_left=mtx_left, dist_left=dist_left, mtx_right=mtx_right, dist_right=dist_right, R=R, T=T)
                    return left_map1, left_map2, right_map1, right_map2, Q
                else:
                    raise Exception(f"Something wen't wrong during the stereo calibration")
            else:
                raise Exception(f"One of the two calibrations went wrong... left: {ret_left} right : {ret_right}")
        else:
            raise Exception(f"One of the two cameras could not see the chessboard... left : {ret_left} right : {ret_right}")


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

    cap_left = cv.VideoCapture(left_id)
    cap_right = cv.VideoCapture(right_id)



    set_cap_property(cap_left)
    set_cap_property(cap_right)

    cap_left.set(cv.CAP_PROP_EXPOSURE, 250)  # Example: -4 = moderate exposure
    cap_right.set(cv.CAP_PROP_EXPOSURE, 250)  # Example: -4 = moderate exposure


    # Prepare object points (e.g., (0,0,0), (1,0,0), ..., (6,8,0))
    objp = np.zeros((obj[0] * obj[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:obj[1], 0:obj[0]].T.reshape(-1, 2)

    left_map1, left_map2, right_map1, right_map2, Q = (None, None, None, None, None)

    while True:

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            raise Exception(f"Something went wrong during frame capture. Left : {ret_left} right: {ret_right}")

        # Combine the frames horizontally
        combined = cv.hconcat([frame_left, frame_right])

        # Display the combined frame
        cv.imshow("Calib", combined)

        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('s'):
            if left_map1 is not None and left_map2 is not None and right_map1 is not None and right_map2 is not None and Q is not None:
                np.savez("calib.npz", left_map1=left_map1, left_map2=left_map2, right_map1=right_map1, right_map2=right_map2, Q=Q)
            else:
                raise Exception(f"One of the value was not none...")
        elif key == ord('c'):
            left_map1, left_map2, right_map1, right_map2, Q = stereo_calibration(frame_left, frame_right)
