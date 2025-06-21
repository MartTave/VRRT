import glob
import os
import sys

import cv2
from ultralytics import YOLO

model = YOLO("./models/fine_tuned/new_dataset/yolo_l/weights/best.pt")

videos = list(sorted(glob.glob(os.path.join("./data/recorded/left", "video_*.mp4"))))

device = sys.argv[2]
device = int(device)

print(f"Working on device : {device}")

for v in videos:
    print(f"Doing for V: {v}")
    results = model.track(v, save_txt=True, classes=[0], verbose=False, device=sys.argv[2], stream=True, vid_stride=3, conf=0.3)
    for r in results:
        annotated = r.plot()
        cv2.imshow("frame", annotated)
        cv2.waitKey(1)
        pass
print("done")
