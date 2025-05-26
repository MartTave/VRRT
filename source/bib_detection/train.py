from ultralytics import YOLO

base_model = YOLO("yolo11n.pt")


results = base_model.train(data="./data/robotflow/data.yaml", epochs=20, imgsz=640, batch=0.8)
