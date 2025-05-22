import cv2
import time

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 25)

print(f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}@{cap.get(cv2.CAP_PROP_FPS)}")

# Warm up
for _ in range(10):
    cap.read()

frame_count = 0
start = time.time()

while frame_count < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

end = time.time()
actual_fps = frame_count / (end - start)
print(f"Actual FPS: {actual_fps:.2f}")
print(f"Took : {end-start:.2f} and should have taken : {frame_count / 25}")
cap.release()
