from ultralytics import YOLO

base_model = YOLO("yolo11n.pt")


results = base_model.train(data="./data/roboflow_2/data.yaml", epochs=100, imgsz=640, batch=0.8, cache=True, workers=64)
