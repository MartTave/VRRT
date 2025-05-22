from classes.capture import get_cap
from classes.track import Tracker
import cv2
cap = cv2.VideoCapture("./data/youtube/run_1.MP4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter("./output/tracked_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
tracker = Tracker()

frame_count = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = tracker.track(frame)
        annotated_frame = tracker.anotate(frame, results)
        writer.write(annotated_frame)
        if frame_count % 100 == 0:
            print(f"Frame {frame_count}/{length}")
        frame_count += 1
    else:
        # Break the loop if the end of the video is reached
        print("End reached")
        break

# Release the video capture object and close the display window
cap.release()
writer.release()
cv2.destroyAllWindows()
