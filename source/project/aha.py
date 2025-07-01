from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model.track("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    import ipdb

    ipdb.set_trace()
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
