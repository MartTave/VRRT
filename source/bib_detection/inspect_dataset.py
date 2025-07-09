import glob
import os

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

DATA_FOLDER = "./data/mit/set1_org"


pics = list(sorted(glob.glob(os.path.join(DATA_FOLDER, "*.JPG"))))
labels = list(sorted(glob.glob(os.path.join(DATA_FOLDER, "*.JPG.mat"))))

for pic, mat in tqdm(zip(pics, labels, strict=True)):
    im = cv2.imread(pic)
    label = scipy.io.loadmat(mat)

    tags = label["tagp"]

    for tag in tags:
        point1 = tag[:2].astype(np.int16)
        point2 = tag[2:].astype(np.int16)
        point2 = [point2[1], point2[0]]
        point1 = [point1[1], point1[0]]
        im = cv2.rectangle(im, point1, point2, (255, 0, 0))
    cv2.imshow("frame", im)
    cv2.waitKey()
