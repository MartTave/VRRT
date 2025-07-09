from ultralytics import YOLO
import cv2

model = YOLO("./weights/best.pt")

results = model("/home/marttave/projects/Bachelor/source/bib_detection/data/robotflow/train/images/0AKY3655_jpg.rf.1bdc1928a9642a7c0f799b8c89c6149d.jpg")

print(results)
annoted = results[0].plot()

cv2.imshow("frame", annoted)
cv2.waitKey()
