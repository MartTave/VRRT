import os

import cv2
from tqdm import tqdm

BASE_FOLDER = "./data/roboflow/train"

# Paths
image_dir = os.path.join(BASE_FOLDER, "images")  # Directory containing images
label_dir = os.path.join(BASE_FOLDER, "labels_updated")  # Directory containing YOLO .txt labels
output_dir = "./visualized_labels"  # Directory to save visualized images
os.makedirs(output_dir, exist_ok=True)

# Class names and colors (modify as needed)
class_names = {0: "person", 1: "bib_number"}
colors = {
    0: (0, 255, 0),  # Green for person
    1: (255, 0, 0),  # Red for bib_number
}


# Process each image
for img_name in tqdm(os.listdir(image_dir)):
    if not img_name.endswith((".jpg", ".png", ".jpeg")):
        continue

    # Load image
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W = img.shape[:2]

    # Load corresponding label file
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        continue

    # Draw bounding boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO normalized coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * W)
            y1 = int((y_center - height / 2) * H)
            x2 = int((x_center + width / 2) * W)
            y2 = int((y_center + height / 2) * H)

            # Draw rectangle and label
            color = colors.get(class_id, (0, 0, 255))  # Default: blue if class not found
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"{class_names.get(class_id, 'unknown')}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # Save or display the image
    output_path = os.path.join(output_dir, f"viz_{img_name}")
    cv2.imwrite(output_path, img)
    # Uncomment to display interactively:
    # cv2.imshow("Label Check", img)
    # cv2.waitKey(0)  # Press any key to continue

print(f"Visualization saved to: {output_dir}")
