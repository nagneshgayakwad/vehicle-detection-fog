from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8m.pt")

def detect_vehicles(img):
    if not isinstance(img, np.ndarray):
        raise ValueError("Invalid image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    annotated = results[0].plot()

    vehicle_classes = ['car', 'bus', 'truck', 'motorbike']
    count = sum(
        1 for r in results[0].boxes.cls
        if model.names[int(r)] in vehicle_classes
    )

    return annotated, count