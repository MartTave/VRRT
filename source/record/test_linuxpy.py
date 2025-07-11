import os
import queue
import threading
import time
from datetime import datetime

import cv2
import pygame
from linuxpy.video.device import Device, VideoCapture

side_dict = {0: "left", 2: "right"}

FPS = 24
RESOLUTION = (1920, 1080)

sync_start = time.time() + 3


class CameraThread(threading.Thread):
    def __init__(self, camera_id, focus_value=None, exposure_value=None):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.focus_value = focus_value
        self.exposure_value = exposure_value
        self.running = False
        self.queue = queue.Queue(maxsize=60)
        self.out = cv2.VideoWriter(f"output_{camera_id}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))

    def run(self):
        self.running = True
        last_time = time.time()

        with Device.from_id(self.camera_id) as cam:
            # Set camera controls
            cam.controls.values()  # This is needed to "load ?" the controls
            if self.focus_value is not None:
                cam.controls["focus_automatic_continuous"] = False  # Disable autofocus
                cam.controls["focus_absolute"] = self.focus_value

            if self.exposure_value is not None:
                cam.controls["auto_exposure"] = 1
                cam.controls["exposure_time_absolute"] = self.exposure_value

            capture = VideoCapture(cam)
            capture.set_format(RESOLUTION[0], RESOLUTION[1], "MJPG")
            capture.set_fps(FPS)
            clock = pygame.time.Clock()
            with capture:
                start = None
                started = False
                for i, frame in enumerate(capture):
                    if not started:
                        started = True
                        time.sleep(sync_start - time.time())
                        start = time.time()
                        print(f"Started at : {start}")
                    if not self.running:
                        print("Exiting...")
                        break

                    clock.tick(FPS)
                    img = cv2.imdecode(frame.array, cv2.IMREAD_COLOR)
                    if img is None:
                        print("Frame not read correctly")
                        continue
                    try:
                        self.queue.put((img, datetime.now().strftime("%H:%M:%S.%f")), timeout=0.1)
                    except queue.Empty:
                        print("Queue is full, skipping frames")
                        continue
            print(f"Took {i} frames in {time.time() - start} time. AVG FPS = {i / (time.time() - start)}")

    def stop(self):
        self.running = False


class WriterThread(threading.Thread):
    def __init__(self, camera1: CameraThread, camera2: CameraThread, frame_per_file=30 * 60 * 5):
        threading.Thread.__init__(self)
        self.queue1 = camera1.queue
        self.queue2 = camera2.queue
        self.folder1 = os.path.join("./results", f"cam_{side_dict[camera1.camera_id]}")
        self.folder2 = os.path.join("./results", f"cam_{side_dict[camera2.camera_id]}")
        self.running = False
        self.frame_per_file = frame_per_file
        os.makedirs(self.folder1, exist_ok=True)
        os.makedirs(self.folder2, exist_ok=True)

    def get_filename(
        self,
    ):
        now = datetime.now()
        filename = f"{now.day:02}.{now.month:02}.{now.year:2}_{now.hour:02}:{now.minute:02}:{now.second:02}.mp4"
        return os.path.join(self.folder1, filename), os.path.join(self.folder2, filename)

    def run(self):
        self.running = True
        index = 0
        curr_filename1, curr_filename2 = self.get_filename()
        writer1 = cv2.VideoWriter(curr_filename1, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESOLUTION)
        writer2 = cv2.VideoWriter(curr_filename2, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESOLUTION)
        black = (0, 0, 0)
        white = (255, 255, 255)
        while self.running:
            try:
                img1, timestamp1 = self.queue1.get(timeout=0.1)
                img2, timestamp2 = self.queue2.get(timeout=0.1)
                img1 = cv2.putText(img1, timestamp1, (10, 20), 2, 0.5, white)
                img2 = cv2.putText(img2, timestamp2, (10, 20), 2, 0.5, white)
                writer1.write(img1)
                writer2.write(img2)
                index += 1
                if index % 100 == 0:
                    filename = os.path.join(self.folder1, f"pic_{index}.png")
                    cv2.imwrite(filename, img1)
                    filename = os.path.join(self.folder2, f"pic_{index}.png")
                    cv2.imwrite(filename, img2)
                if index == self.frame_per_file:
                    index = 0
                    curr_filename1, curr_filename2 = self.get_filename()
                    writer1.release()
                    writer2.release()
                    writer1 = cv2.VideoWriter(curr_filename1, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESOLUTION)
                    writer2 = cv2.VideoWriter(curr_filename2, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RESOLUTION)
            except queue.Empty:
                continue
        writer1.release()
        writer2.release()
        print("Writer thread stopping.")

    def stop(self):
        self.running = False


if __name__ == "__main__":
    cam1 = CameraThread(2, focus_value=50, exposure_value=100)
    cam2 = CameraThread(0, focus_value=60, exposure_value=150)
    writer = WriterThread(cam1, cam2)
    try:
        writer.start()
        cam1.start()
        cam2.start()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping cameras...")
        cam1.stop()
        writer.stop()
        cam2.stop()
        cam1.join()
        writer.join()
        cam2.join()
        print("Done.")
