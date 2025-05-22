import cv2
import numpy as np
from boxmot import ByteTrack
from ultralytics import YOLO
import torch

def track_persons_in_video(input_path, output_path, conf_threshold=0.3):

    device = torch.device('cpu')

    # Initialize YOLO model for person detection
    model = YOLO('yolo11n.pt')  # Using YOLOv8 nano model (you can use m/l/x for better accuracy)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the tracker
    tracker = ByteTrack(frame_rate=int(round(fps)), track_buffer=250)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection (only for person class - class 0)
        results = model(frame, classes=[0], verbose=False)  # Only detect persons

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())

                if confidence > conf_threshold:
                    detections.append([x1, y1, x2, y2, confidence])

        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        # Update tracker with new detections

        detections = np.array([[*d, 0] for d in detections])
        tracks = tracker.update(detections, frame)

        # Draw tracks on frame
        for track in tracks:
            # Extract track information
            bbox = track[:4].astype(int)
            track_id = int(track[4])

            x1, y1, x2, y2 = bbox

            # Generate unique color for this ID
            color = ((track_id * 123) % 255, (track_id * 50) % 255, (track_id * 100) % 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID label
            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Write the frame to output video
        out.write(frame)

        # Display progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    input_video = "../../data/GX030012.MP4"  # Change this to your input video path
    output_video = "output_tracked.mp4"
    track_persons_in_video(input_video, output_video)
