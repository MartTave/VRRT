import numpy as np
import yaml


def create_camera_info_yaml(file_name, resolution):
    file = np.load(f"calibrations/{file_name}")

    data = {
        "image_width": width,
        "image_height": height,
        "camera_name": camera_name,
        "distortion_model": "plumb_bob",
        "D": D.flatten().tolist(),
        "K": K.flatten().tolist(),
        "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Identity
        "P": P.flatten().tolist(),
    }
    with open(f"{camera_name}.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=True)


# Example usage (replace with your actual values)
K_left = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # Example intrinsics
D_left = np.array([-0.1, 0.01, 0, 0, 0])  # Example distortion
P1 = np.array([[500, 0, 320, -50], [0, 500, 240, 0], [0, 0, 1, 0]])  # Example P1

create_camera_info_yaml("left_camera", K_left, D_left, P1, 640, 480)
