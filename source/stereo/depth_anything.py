from transformers import pipeline
from PIL import Image
import requests
import cv2
import numpy
import torch
import gc

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)

cap = cv2.VideoCapture("./video_24.05.2025_19:11:25.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()

print(frame.shape[:2])

writer = cv2.VideoWriter("./result/res.mp4", fourcc=cv2.VideoWriter.fourcc(*'MJPG'), fps=fps, frameSize=(frame.shape[1], frame.shape[0]))

frames = []

BATCH = 512

frame_i = 0

while ret:
    frame_i += 1
    ret, frame = cap.read()

    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if len(frames) == BATCH:
        print("Doing a batch !")
        results = pipe(frames, batch_size=BATCH)
        print("Batch done !")
        if results is None:
            continue
        for res in results:
            depth_map = cv2.cvtColor(numpy.array(res["depth"]), cv2.COLOR_RGB2BGR)
            writer.write(depth_map)
        torch.cuda.empty_cache()
        gc.collect()
        frames = []

# frame_i = 0
# while True:
#     frame_i += 1
#     ret, frame = cap.read()

#     if not ret:
#         break

#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     res = pipe(img)
#     to_print = cv2.cvtColor(numpy.array(res["depth"]),cv2.COLOR_RGB2BGR)
#     writer.write(to_print)
#     if frame_i % 100 == 0:
        # print(f"At frame {frame_i}")


cap.release()
writer.release()
