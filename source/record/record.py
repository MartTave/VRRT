from datetime import datetime
import os
import cv2
import threading
import queue
import time

# Config
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FPS = 25
FOLDER = './videos'

# Shared queue for frames
frame_queue = queue.Queue(maxsize=100)  # Adjust size depending on memory
stop_event = threading.Event()  # Signal for clean shutdown

def get_capture(id):
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap

def camera_reader():
    cap = get_capture(2)

    for _ in range(20):
        ret, _ = cap.read()
        assert ret

    start = time.time()
    frame_n = 0
    dropped = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        frame_n += 1
        try:
            frame_queue.put(frame, timeout=1/TARGET_FPS)
        except queue.Full:
            dropped += 1
            print("Frame queue full - dropping frame")
    end = time.time()
    print(f"Took {frame_n} frames in {end-start:.2f}s. Avg : {frame_n/(end-start)}FPS")
    print(f"Dropped frames : {dropped}")
    cap.release()

def get_filename(prefix="video"):
    now = datetime.now()
    return f"{prefix}_{now.day:02}.{now.month:02}.{now.year:2}_{now.hour:02}:{now.minute:02}:{now.second:02}.mp4"

# Thread to write frames to file
def video_writer():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(FOLDER, get_filename()), fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_n = TARGET_FPS * 60 * 1
    for i in range(frame_n):
        if stop_event.is_set() and frame_queue.empty():
            break
        frame = frame_queue.get()
        out.write(frame)

    out.release()


if __name__ == "__main__":
    # Start threads
    reader_thread = threading.Thread(target=camera_reader, daemon=True)
    reader_thread.start()

    writer_thread = None

    # Keep main thread alive
    try:

        while True:
            writer_thread = threading.Thread(target=video_writer, daemon=True)
            writer_thread.start()
            writer_thread.join()
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
        reader_thread.join()
        writer_thread.join()
        print("All threads stopped cleanly.")
