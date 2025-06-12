import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2

# Shared queue for frames
frame_queue = queue.Queue(maxsize=100)  # Adjust size depending on memory
stop_event = threading.Event()  # Signal for clean shutdown


def get_capture(id, frame_width, frame_height, fps):
    cap = cv2.VideoCapture(id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


def camera_reader(target_id, frame_width, frame_height, fps):
    cap = get_capture(target_id, frame_width=frame_width, frame_height=frame_height, fps=fps)

    for _ in range(20):
        ret, _ = cap.read()
        assert ret

    start = time.time()
    frame_n = 0
    dropped = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        frame_timestamp = time.time()
        frame_n += 1
        try:
            frame_queue.put((frame, frame_timestamp), timeout=1 / fps)
        except queue.Full:
            dropped += 1
            print("Frame queue full - dropping frame")
    end = time.time()
    print(f"Took {frame_n} frames in {end - start:.2f}s. Avg : {frame_n / (end - start)}FPS")
    print(f"Dropped frames : {dropped}")
    cap.release()


def get_filename(prefix="video"):
    now = datetime.now()
    return (
        f"{prefix}_{now.day:02}.{now.month:02}.{now.year:2}_{now.hour:02}:{now.minute:02}:{now.second:02}.mp4",
        f"{prefix}_{now.day:02}.{now.month:02}.{now.year:2}_{now.hour:02}:{now.minute:02}:{now.second:02}.txt",
    )


def get_formatted_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S.%f")[:-3]  # Truncate last 3 digits to get milliseconds


# Thread to write frames to file
def video_writer(dest_folder, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(dest_folder, exist_ok=True)
    video_filename, text_filename = get_filename()
    video_filepath = os.path.join(dest_folder, video_filename)
    text_filepath = os.path.join(dest_folder, text_filename)
    out = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))
    file = open(text_filepath, "w")
    frames_per_minute = fps * 60
    frame_n = frames_per_minute * 5
    start = time.time()
    i = 0
    while i < frame_n:
        if stop_event.is_set() and frame_queue.empty():
            break
        try:
            frame, timestamp = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        i += 1
        out.write(frame)
        file.write(f"{i};{timestamp}\n")
        if i % frames_per_minute == 0:
            now = time.time()
            time_taken = now - start
            print(f"Recording at {i / frames_per_minute:.0f}min of {frame_n / frames_per_minute:.0f}min")
            print(f"Took {time_taken:.2f}")
            start = now

    print(f"Written file {video_filepath} and {text_filepath}")
    out.release()


def video_viewer():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    # Start threads

    def set_args_parser():
        # Shared arguments (not the main parser)
        shared_parser = argparse.ArgumentParser(add_help=False)
        shared_parser.add_argument("--camera", type=int, help="The camera stream to capture", required=True)
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

    reader_thread = threading.Thread(target=camera_reader, args=(args.camera, args.width, args.height, args.fps), daemon=True)
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
