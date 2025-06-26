import time
from glob import glob

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def crop_bottom_right(image, new_width, new_height):
    height, width = image.shape[:2]
    x = width - new_width
    y = height - new_height
    return image[y:height, x:width]


DEVICE = 0

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

frames = []
files = sorted(list(glob("./data/dataset/pic_*.png")))[:50]
for f in files:
    frame = cv2.imread(f)
    frames.append(crop_bottom_right(frame, 1280, 720))

encoder = "vits"  # or 'vits', 'vitb', 'vitg'
for encoder in ["vits", "vitb"]:
    for size in [(128, 72), (256, 144), (384, 216), (512, 288), (640, 360), (1280, 720)]:
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        model = model.to(DEVICE).eval()

        images = []
        then1 = time.time()

        for f in frames:
            image, (h, w) = model.image2tensor(f)
            images.append((image, h, w))
        print("Tensor created !")
        then2 = time.time()
        depths = []
        for i in images:
            depths.append(model.infer_image_cuda(*i))

        print(f"{encoder} --- {size} : ")
        print(f"Done {len(depths)} in {time.time() - then2} avg : {len(depths) / (time.time() - then2)} FPS")

        for i, depth in enumerate(depths):
            if i == 27:
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                frame = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                cv2.imwrite(f"./test/{encoder}_{size[0]}x{size[1]}.png", frame)
