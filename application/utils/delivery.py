import cv2
import easyocr
from ultralytics import YOLO
import os
from datetime import datetime
from ..utils.utils import cod_detection_labels
# Load YOLO model
base_dir = os.path.dirname(os.path.abspath(__file__))

# Model path for score-detector.pt, used for primary score detection
model_path = os.path.join(base_dir, '..', '..', 'model', 'score-detector-2.pt')

model = YOLO(model_path)
reader = easyocr.Reader(['en'])

def filter_score(text):
    replacements = {
        'o': '0', 'O': '0', 'Q': '0', 'D': '0',
        'l': '1', 'I': '1', 'J': '1', "j": "1", 'i': '1', '|': '1',
        'Z': '2', 'z': '2', 'S': '5',
        'B': '8', 'g': '9', 'q': '9', 'P': '9',
        'G': '6', 'b': '6', 's': '5',
        'A': '4', 'R': '2', 'T': '7', 'Y': '7',
        'W': '11', 'M': '11', 'd': '4', 'F': '16'
    }
    return ''.join(replacements.get(c, c) for c in text)

# Global variables
video_path = ""
cap = None
is_running = False
scores_data = []  # List to store scores
start_time = None  # To track the start time of inference

# Create a folder to save CSV files
if not os.path.exists("scores_data"):
    os.makedirs("scores_data")

def process_video(frame):
    global is_running, cap, scores_data
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