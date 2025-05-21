import cv2
from ultralytics import YOLO
import os
from ..utils.utils import number_detection_labels
# Load YOLO model
base_dir = os.path.dirname(os.path.abspath(__file__))

# Model path for score-detector.pt, used for primary score detection
model_path = os.path.join(base_dir, '..', '..', 'model', 'number-detector.pt')

model = YOLO(model_path)

def number_detector(frame):
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

        label = number_detection_labels.get(cls, 'unknown')

        detected_regions[label] = resized_image

    return detected_regions