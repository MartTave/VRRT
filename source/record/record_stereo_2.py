import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2

# Shared queue for frames
frame_queue = queue.Queue(maxsize=100)
frame_queue_left = queue.Queue(maxsize=1)  # Adjust size depending on memory
frame_queue_right = queue.Queue(maxsize=1)  # Adjust size depending on memory
stop_event = threading.Event()  # Signal for clean shutdown

cam_dict = {
    0: "left",
    2: "right",
}


def get_capture(id, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Deactivating buffer to allow for syncronized frame capture
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    return cap


def frame_in_queue(capture, queue):
    while not stop_event.is_set():
        ret, frame = capture.retrieve()
        if not ret:
            time.sleep(1 / 60)
            continue
        queue.put(frame)


def camera_reader(camera_id, queue, frame_width, frame_height, fps):
    capture = get_capture(camera_id, frame_width=frame_width, frame_height=frame_height, fps=fps)

    for _ in range(20):
        ret, _ = capture.read()

    start = time.time()
    frame_n = 0
    dropped = 0

    while not stop_event.is_set():
        start = time.time()
        _, frame1 = capture.read()
        timestamp1 = datetime.now().strftime("%H:%M:%S.%f")
        frame_n += 1
        try:
            frame_queue.put((frame1, timestamp1, frame2, timestamp2), timeout=1 / fps)
        except queue.Full:
            dropped += 1
            print("Frame queue full - dropping frame")
        if time.time() - start > 1 / fps:
            print(f"Excedding time budget. Took : {time.time() - start}")
    end = time.time()
    print(f"Took {frame_n} frames in {end - start:.2f}s. Avg : {frame_n / (end - start)}FPS")
    print(f"Dropped frames : {dropped}")
    cap_left.release()
    cap_right.release()


def write_timestamp(image, timestamp):
    color = (255, 0, 0)
    return cv2.putText(image, timestamp, (10, 20), 2, 0.5, color)


def get_filename(prefix="video"):
    now = datetime.now()
    return f"{prefix}_{now.day:02}.{now.month:02}.{now.year:2}_{now.hour:02}:{now.minute:02}:{now.second:02}.mp4"


# Thread to write frames to file
def video_writer(dest_folder, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    folder_left = os.path.join(dest_folder, "camera_left")
    folder_right = os.path.join(dest_folder, "camera_right")
    os.makedirs(folder_left, exist_ok=True)
    os.makedirs(folder_right, exist_ok=True)
    filepath_left = os.path.join(folder_left, get_filename())
    filepath_right = os.path.join(folder_right, get_filename())
    out_left = cv2.VideoWriter(filepath_left, fourcc, fps, (frame_width, frame_height))
    out_right = cv2.VideoWriter(filepath_right, fourcc, fps, (frame_width, frame_height))
    frames_per_minute = fps * 60
    frame_n = frames_per_minute * 5
    start = time.time()
    for i in range(frame_n):
        if stop_event.is_set() and frame_queue.empty():
            break
        try:
            frame_left, timestamp_left, frame_right, timestamp_right = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame_left = write_timestamp(frame_left, timestamp_left)
        frame_right = write_timestamp(frame_right, timestamp_right)
        out_left.write(frame_left)
        out_right.write(frame_right)
        if i % frames_per_minute == 0:
            now = time.time()
            time_taken = now - start
            print(f"Recording at {i / frames_per_minute:.0f}min of {frame_n / frames_per_minute:.0f}min")
            print(f"Took {time_taken:.2f}")
            start = now

    print(f"Written file {filepath}")
    out_left.release()
    out_right.release()


def video_viewer():
    while not stop_event.is_set():
        try:
            frame_left, timestamp_left, frame_right, timestamp_right = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame_left = write_timestamp(frame_left, timestamp_left)
        frame_right = write_timestamp(frame_right, timestamp_right)
        concated = cv2.hconcat((frame_left, frame_right))
        cv2.imshow("frame", concated)
        cv2.waitKey(1)


if __name__ == "__main__":
    # Start threads

    def set_args_parser():
        # Shared arguments (not the main parser)
        shared_parser = argparse.ArgumentParser(add_help=False)
        shared_parser.add_argument("--camera-left", type=int, help="The camera stream to capture", required=True)
        shared_parser.add_argument("--camera-right", type=int, help="The camera stream to capture", required=True)
        shared_parser.add_argument("--fps", type=int, help="The FPS at which to capture the stream", default=30)
        shared_parser.add_argument("--width", type=int, help="The width of the frame to capture", default=640)
        shared_parser.add_argument("--height", type=int, help="The height of the frame to capture", default=480)

        # Main parser (top-level)
        main_parser = argparse.ArgumentParser()
        subparser = main_parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

        # Subcommands
        record_parser = subparser.add_parser("record", parents=[shared_parser], help="Record camera stream")
        record_parser.add_argument("--output", type=str, required=True, help="The output folder for the videos")

        preview_parser = subparser.add_parser("preview", parents=[shared_parser], help="Preview camera stream")

        return main_parser

    main_parser = set_args_parser()

    # Parse
    args = main_parser.parse_args()

    reader_thread = threading.Thread(target=camera_reader, args=(args.camera_left, args.camera_right, args.width, args.height, args.fps), daemon=True)
    reader_thread.start()

    if args.mode == "record":
        writer_thread = None

        # Keep main thread alive
        try:
            while True:
                writer_thread = threading.Thread(target=video_writer, args=(args.output, args.fps, args.width, args.height), daemon=True)
                writer_thread.start()
                writer_thread.join()
        except KeyboardInterrupt:
            print("Stopping...")
            stop_event.set()
            reader_thread.join()
            if writer_thread is not None:
                writer_thread.join()
            print("All threads stopped cleanly.")
    elif args.mode == "preview":
        preview_thread = None
        try:
            preview_thread = threading.Thread(target=video_viewer, daemon=True)
            preview_thread.start()
            preview_thread.join()
        except KeyboardInterrupt:
            print("Stopping...")
            stop_event.set()
            reader_thread.join()
            if preview_thread is not None:
                preview_thread.join()
            print("All thread stopped cleanly.")
