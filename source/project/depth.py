import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def crop_bottom_right(image, new_width, new_height):
    height, width = image.shape[:2]
    x = width - new_width
    y = height - new_height
    return image[y:height, x:width]


class ArrivalLine:
    def __init__(self, encoder="vits", target_depth=3, reversed=True):
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        self.reversed = reversed

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
        self.model.to(torch.device(0))
        self.persons_depth = {}
        self.target_depth = target_depth
        pass

    def new_frame(self, frame, person_boxes):
        if person_boxes.id is None:
            return []

        def get_box_center(xyxy):
            return int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)

        depth = self.model.infer_image(frame)
        arrived = []
        for box, id in zip(person_boxes.xyxy, person_boxes.id, strict=False):
            if id not in self.persons_depth.keys():
                self.persons_depth[id] = {
                    "depths": np.array([]),
                    "arrived": False,
                }
            if self.persons_depth[id]["arrived"]:
                continue
            center = get_box_center(box)

            curr_person_depth = depth[center[0]][center[1]]
            self.persons_depth[id]["depths"] = np.append(self.persons_depth[id]["depths"], [curr_person_depth])
            if len(self.persons_depth[id]["depths"]) > 10:
                self.persons_depth[id]["depths"] = np.delete(self.persons_depth[id]["depths"], [0])
            avg = np.average(self.persons_depth[id]["depths"])
            if curr_person_depth < avg and self.reversed or curr_person_depth > avg and not self.reversed:
                # The person is moving in the right direction
                if (curr_person_depth < self.target_depth and self.reversed) or (curr_person_depth > self.target_depth and not self.reversed):
                    self.persons_depth[id]["arrived"] = True
                    arrived.append(id)

        return arrived
