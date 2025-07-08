import logging

import cv2


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        message = super().format(record)
        return f"{color}{message}{reset}"


def annotate_box(frame, boxes, basepoint=(0, 0), color=(255, 0, 0)):
    if boxes is None:
        return frame
    for b in boxes.xyxy:
        x1, y1, x2, y2 = b
        x1 = int(x1 + basepoint[0])
        y1 = int(y1 + basepoint[1])
        x2 = int(x2 + basepoint[0])
        y2 = int(y2 + basepoint[1])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=1)
    return frame


def crop(frame, points):
    return frame[points[0][1] : points[1][1], points[0][0] : points[1][0]]


def get_colored_logger(name):
    logger = logging.getLogger(name)

    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = ColoredFormatter("[%(name)s] %(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def crop_from_boxes(frame, boxes):
    res = []
    for b in boxes.xyxy:
        res.append(frame[b[1].int() : b[3].int(), b[0].int() : b[2].int()].copy())
    return res


ref_points = []


def click_and_crop(frame):
    clone = frame.copy()

    def click_handler(event, x, y, flags, param):
        global ref_points

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_points = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            ref_points.append((x, y))
            # Draw rectangle on the image
            frame = clone.copy()
            cv2.rectangle(frame, ref_points[0], ref_points[1], (0, 255, 0), 2)
            cv2.imshow("Cropping window", frame)

    cv2.namedWindow("Cropping window")
    cv2.setMouseCallback("Cropping window", click_handler)
    cv2.imshow("Cropping window", frame)
    while True:
        key = cv2.waitKey(1)

        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    res = []
    minX = ref_points[0][0]
    maxX = ref_points[0][0]
    minY = ref_points[0][1]
    maxY = ref_points[0][1]
    for r in ref_points:
        if r[0] < minX:
            minX = r[0]
        elif r[0] > maxX:
            maxX = r[0]

        if r[1] < minY:
            minY = r[1]
        elif r[1] > maxY:
            maxY = r[1]
    return ((minX, maxX), (minY, maxY))
