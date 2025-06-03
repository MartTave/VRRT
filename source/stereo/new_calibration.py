import os
import cv2
import glob
from cv2 import aruco

def calibrate_from_pics(path):
    filenames = os.listdir(path)

    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50))

    board = aruco.CharucoBoard((8, 13), 21, 15, aruco.getPredefinedDictionary(aruco.DICT_4X4_50))

    charuco_detector = aruco.CharucoDetector(board)

    for f in filenames:
        assert f.endswith("*.png")

        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)

        corners, ids, _ = detector.detectMarkers(img)

        if len(corners) > 10:
            ret, charuco_corners, charuco_ids = aruco.CharucoDetector
        else:
            print("Not enough corners found ! skipping.")
