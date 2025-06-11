import datetime
import threading
import time

import cv2
from linuxpy.video.device import Device, VideoCapture


class CameraThread(threading.Thread):
    def __init__(self, camera_id, focus_value=None, exposure_value=None):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.focus_value = focus_value
        self.exposure_value = exposure_value
        self.running = False
        self.out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))

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
            capture.set_format(640, 480, "MJPG")
            capture.set_fps(30)

            with capture:
                for i, frame in enumerate(capture):
                    if not self.running:
                        break

                    # Get precise timestamp (monotonic clock is best for intervals)
                    img = cv2.imdecode(frame.array, cv2.IMREAD_COLOR)
                    img = cv2.putText(img, datetime.datetime.now().strftime("%H:%M:%S.%f"), (10, 20), 2, 0.5, (0, 0, 0))
                    self.out.write(img)  # Write frame to file

    def stop(self):
        self.running = False


# Example usage
if __name__ == "__main__":
    # Create camera threads with desired settings
    # Parameters: camera_id, focus_value (0-255 or None for auto), exposure_value
    cam1 = CameraThread(2, focus_value=50, exposure_value=100)
    # cam2 = CameraThread(4, focus_value=60, exposure_value=150)

    try:
        # Start cameras
        cam1.start()
        # cam2.start()

        # Keep main thread alive
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping cameras...")
        cam1.stop()
        # cam2.stop()
        cam1.join()
        # cam2.join()
        print("Done.")
