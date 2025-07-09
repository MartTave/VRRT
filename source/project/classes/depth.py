import cv2
import numpy as np
import torch
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from classes.depth_anything_v2.dpt import DepthAnythingV2


def crop_bottom_right(image, new_width, new_height):
    height, width = image.shape[:2]
    x = width - new_width
    y = height - new_height
    return image[y:height, x:width]


def points_not_in_any_box(points, boxes):
    """Filter points that lie outside all given boxes.

    Args:
        points: np.array of shape (n, 2) containing 2D points
        boxes: List of tuples, each tuple is (x_min, y_min, x_max, y_max)

    Returns:
        np.array of points that lie outside all boxes

    """
    if len(points) == 0:
        return np.empty((0, 2))
    if len(boxes) == 0:
        return points.copy()  # No boxes → all points are outside

    # Start with all points included
    mask = np.ones(len(points), dtype=bool)

    for box in boxes:
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        # Remove points that are inside this box
        mask &= (points[:, 0] < x_min) | (points[:, 0] > x_max) | (points[:, 1] < y_min) | (points[:, 1] > y_max)

    return points[mask]


# By default, with depth anything.
# Far object have a lower score, closest get higher !
# so if the run is from far to close, reversed needs to be at false

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def get_arrival_line(picture):
    points = []
    window_name = "arrival selection"
    drawed = picture.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point to our list

            if len(points) == 2:
                points.pop()
                points.pop()

            points.append((x, y))
            # Draw a small circle at the click position

            # If we have at least 2 points, draw a line between the last two
            drawed = picture.copy()
            if len(points) >= 2:
                cv2.line(drawed, points[0], points[1], (0, 255, 0), 2)
            for p in points:
                cv2.circle(drawed, p, 5, (0, 0, 255), -1)

            # Update the display
            cv2.imshow(window_name, drawed)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, drawed)
    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            raise Exception("Q pressed, interupting")
        elif key == ord("s"):
            break
    cv2.destroyWindow(window_name)
    assert len(points) == 2
    return points


class ArrivalLine:
    def __init__(self, line: dict, device=torch.device(0), encoder="vits", reversed=False, min_slope=1e-2):
        self.loaded_frames = []
        self.reversed = reversed

        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{encoder}.pth", map_location=device))
        self.model.to(device)
        self.persons_depth = {}
        self.min_slope = min_slope
        self.line_points = (line["start"], line["end"])
        self.line = np.array([(k, v) for k, v in ArrivalLine.get_line_pixels(self.line_points).items()])

    @staticmethod
    def get_line_pixels(points: tuple[tuple, tuple]) -> dict[int, int]:
        start_point, end_point = points
        x1, y1 = start_point
        x2, y2 = end_point

        # Calculate differences
        dx = x2 - x1
        dy = y2 - y1

        # Determine the number of steps needed
        steps = max(abs(dx), abs(dy))

        # Avoid division by zero
        if steps == 0:
            return [(x1, y1)]

        # Calculate increments
        x_inc = dx / steps
        y_inc = dy / steps

        # Generate all points along the line
        line_pixels = {}
        for i in range(steps + 1):
            x = round(x1 + i * x_inc)
            y = round(y1 + i * y_inc)
            line_pixels[x] = y

        return line_pixels

    def treat_depth(self, depth, person_result, frame, annotate=False):
        person_boxes = person_result.boxes
        keypoints = person_result.keypoints
        if person_boxes.id is None:
            print("No id for person boxes !")
            return []

        def get_box_center(xyxy):
            return int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)

        arrived = []

        for i, (box, id) in enumerate(zip(person_boxes.xyxy, person_boxes.id, strict=False)):
            id = int(id)
            color = (0, 0, 255)
            poi = ()
            if keypoints is None:
                poi = get_box_center(box)
            else:
                curr_keypoints = keypoints.xy[i].cpu().numpy().astype(np.int32)
                left_foot = curr_keypoints[15]
                right_foot = curr_keypoints[16]
                if depth[left_foot[1]][left_foot[0]] > depth[right_foot[1]][right_foot[1]]:
                    poi = (left_foot[0], left_foot[1])
                else:
                    poi = (right_foot[0], right_foot[1])
            if id not in self.persons_depth.keys():
                self.persons_depth[id] = {
                    "depths": np.array([]),
                    "arrived": False,
                }
            if self.persons_depth[id]["arrived"]:
                color = (0, 255, 0)
            elif box[2] > frame.shape[1] - 2 or box[0] < 2:
                continue
            else:
                curr_person_depth = depth[poi[1]][poi[0]]

                self.persons_depth[id]["depths"] = np.append(self.persons_depth[id]["depths"], [curr_person_depth])
                if len(self.persons_depth[id]["depths"]) > 120:
                    self.persons_depth[id]["depths"] = np.delete(self.persons_depth[id]["depths"], [0])
                elif len(self.persons_depth[id]["depths"]) < 30:
                    continue
                res = LinearRegression().fit(np.array(range(len(self.persons_depth[id]["depths"]))).reshape(-1, 1), self.persons_depth[id]["depths"])
                # res = linregress(range(len(self.persons_depth[id]["depths"])), self.persons_depth[id]["depths"])
                if np.isnan(res.coef_[0]):
                    continue
                if res.coef_[0] < 0 - self.min_slope and self.reversed or res.coef_[0] > self.min_slope and not self.reversed:
                    # The person is moving in the right direction

                    # We check if the POI is in the right x coordinates
                    if poi[0] in self.line[:, 0]:
                        valid_line_pixels = points_not_in_any_box(self.line.copy(), person_boxes.xyxy)
                        X = valid_line_pixels[:, [0]]
                        y = depth[valid_line_pixels[:, 1], valid_line_pixels[:, 0]]
                        regression = LinearRegression().fit(X, y)
                        curr_target = regression.predict(np.array(poi[0]).reshape(1, -1))[0]
                        if (curr_person_depth < curr_target and self.reversed) or (curr_person_depth > curr_target and not self.reversed):
                            self.persons_depth[id]["arrived"] = True
                            arrived.append(id)
                            color = (0, 255, 0)
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

            if annotate:
                cv2.circle(frame, poi, 7, color, cv2.FILLED)
        return arrived

    def treat_depths(self, depths, persons_results, frames, annotate=False):
        zipped = zip(depths, persons_results, frames, strict=True)
        result = []
        for depth, persons_result, frame in zipped:
            result.append(self.treat_depth(depth, persons_result, frame, annotate))
        return result

    def new_frame(self, frame, person_boxes, annotate=False):
        if person_boxes.id is None:
            print("No id for person boxes !")
            return []

        def get_box_center(xyxy):
            return int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)

        depth = self.model.infer_image(frame)
        arrived = []
        for box, id in zip(person_boxes.xyxy, person_boxes.id, strict=False):
            id = int(id)
            color = (0, 0, 255)
            if id not in self.persons_depth.keys():
                self.persons_depth[id] = {
                    "depths": np.array([]),
                    "arrived": False,
                }
            if self.persons_depth[id]["arrived"]:
                color = (0, 255, 0)
                continue
            center = get_box_center(box)
            curr_person_depth = depth[center[1]][center[0]]
            self.persons_depth[id]["depths"] = np.append(self.persons_depth[id]["depths"], [curr_person_depth])
            if len(self.persons_depth[id]["depths"]) > 120:
                self.persons_depth[id]["depths"] = np.delete(self.persons_depth[id]["depths"], [0])
            elif len(self.persons_depth[id]["depths"]) < 30:
                continue
            res = linregress(range(len(self.persons_depth[id]["depths"])), self.persons_depth[id]["depths"])
            if np.isnan(res.slope):
                continue

            if res.slope < 0 - self.min_slope and self.reversed or res.slope > self.min_slope and not self.reversed:
                # The person is moving in the right direction
                if center[0] in self.line.keys():
                    curr_target = depth[self.line[center[0]]][center[0]]
                    if (curr_person_depth < curr_target and self.reversed) or (curr_person_depth > curr_target and not self.reversed):
                        self.persons_depth[id]["arrived"] = True
                        arrived.append(id)
                        color = (0, 255, 0)
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)

            if annotate:
                cv2.circle(frame, center, 7, color, cv2.FILLED)
        if annotate:
            x = sorted(list(self.line.keys()))
            x_start = x[0]
            x_end = x[-1]
            cv2.line(frame, (x_start, self.line[x_start]), (x_end, self.line[x_end]), (255, 255, 0), 3)
        return arrived
