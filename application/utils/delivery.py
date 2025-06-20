import cv2
import torch
import os
import requests
import tempfile
from ultralytics import YOLO
from ..utils.utils import number_detection_labels, cod_detection_labels

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
    if os.getenv("ENV") == "development":
        hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hugging_face_token:
            print("Missing HUGGING_FACE_TOKEN environment variable")
            return None

        _, buffer = cv2.imencode('.jpg', frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            tmp.write(buffer.tobytes())
            tmp.seek(0)

            headers = {
                "Authorization": f"Bearer {hugging_face_token}"
            }

            response = requests.post(
                "https://sleeplessgamers-llm-ocr.hf.space/api/chat",
                files={"image_file": tmp},
                data={
                    "system_prompt_key": "ocr_digit_only",
                    "stream": "false"
                },
                headers=headers
            )

            if response.ok:
                try:
                    data = response.json()
                    try:
                        predicted_number = int(data["content"])
                    except ValueError as e:
                        if "invalid literal for int()" in str(e):
                            import re
                            digits = re.sub(r"\D", "", data["content"])
                            predicted_number = int(digits) if digits else None
                        else:
                            raise
                    return predicted_number
                except Exception as e:
                    print(f"API parsing error: {e}")
                    return None

            else:
                print(f"OCR API error: {response.status_code}, {response.text}")
                return None

