import glob
import os

import cv2
import numpy as np

input_path = "./data/pics/stereo/"
output_path = "./data/pics/stereo_rectified"
calib_path = "./calibrations/calib_14.06.npz"

left_pics = list(sorted(glob.glob(os.path.join(input_path, "left_*.png"))))
right_pics = list(sorted(glob.glob(os.path.join(input_path, "right_*.png"))))

calib_file = np.load(calib_path)

map_left_x, map_left_y = (calib_file["map_left_x"], calib_file["map_left_y"])
map_right_x, map_right_y = (calib_file["map_right_x"], calib_file["map_right_y"])

for l_pic, r_pic in zip(left_pics, right_pics):
    img_left, img_right = cv2.imread(l_pic), cv2.imread(r_pic)
    rect_img_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_img_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    left_filename, right_filename = os.path.basename(l_pic), os.path.basename(r_pic)

    cv2.imwrite(os.path.join(output_path, left_filename), rect_img_left)
    cv2.imwrite(os.path.join(output_path, right_filename), rect_img_right)
