
from classes.capture import get_cap
from classes.track import Tracker
import cv2
cap = get_cap(2)
tracker = Tracker()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = tracker.track(frame)
        annotated_frame = tracker.anotate(frame, results)
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        print("End reached")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
