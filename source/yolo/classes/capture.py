import cv2

def get_cap(device_id, logitec=False):
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Disable auto-exposure (0 = manual, 1 = auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    cap.set(cv2.CAP_PROP_FPS, 30)

    if logitec:
       cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
       cap.set(cv2.CAP_PROP_FOCUS, 0)  # 50 = mid-range focus


    return cap
