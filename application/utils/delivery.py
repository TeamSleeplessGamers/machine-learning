import cv2
import torch
import os
import time
from ultralytics import YOLO
from ..utils.utils import number_detection_labels, cod_detection_labels, log_time

base_dir = os.path.dirname(os.path.abspath(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = os.path.join(base_dir, '..', '..', 'model', 'warzone.pt')
model_path_2 = os.path.join(base_dir, '..', '..', 'model', 'ocr-detector.pt')

model = YOLO(model_path).to(device)
model_2 = YOLO(model_path_2).to(device)

def process_video(frame):
    results = model(frame, verbose=False)

    class_detections = {}
    detected_regions = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls not in class_detections or conf > class_detections[cls]['conf']:
                class_detections[cls] = {'conf': conf, 'box': (x1, y1, x2, y2)}

    for cls, detection in class_detections.items():
        x1, y1, x2, y2 = detection['box']
        detected_region = frame[y1:y2, x1:x2]

        height_roi = detected_region.shape[0]
        bottom_half = detected_region[int(height_roi // 3): height_roi, :]

        resized_image = cv2.resize(bottom_half, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

        label = cod_detection_labels.get(cls, 'unknown')
        detected_regions[label] = resized_image

    return detected_regions

def number_detector_2(frame):
    results = model_2(frame, verbose=False)

    class_detections = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls not in class_detections or conf > class_detections[cls]['conf']:
                class_detections[cls] = {'conf': conf, 'box': (x1, y1, x2, y2)}

    if not class_detections:
        return None

    most_confident_cls = max(class_detections.items(), key=lambda item: item[1]['conf'])[0]
    label_str = number_detection_labels.get(most_confident_cls, 'unknown')

    try:
        return int(label_str) + 1
    except ValueError:
        return None
