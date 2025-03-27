import cv2
import easyocr
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLO model
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, '..', '..', 'model', 'score-detector.pt')

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

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_video(frame):
    global is_running, cap, scores_data
    prev_score1, prev_score2 = "0", "0"

    results = model(frame, verbose=False)
    extracted_scores = []
    class_detections = {}

    selected_contrast_images = {}  # Store images for each class

    # Step 1: Identify best detections per class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls not in class_detections or conf > class_detections[cls]['conf']:
                class_detections[cls] = {'conf': conf, 'box': (x1, y1, x2, y2)}

    # Step 2: Process detected regions
    for cls, detection in class_detections.items():
        x1, y1, x2, y2 = detection['box']
        detected_region = frame[y1:y2, x1:x2]
        height_roi = detected_region.shape[0]
        bottom_half = detected_region[int(height_roi // 3): height_roi, :]

        resized_image = cv2.resize(bottom_half, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

        # Apply preprocessing for better OCR accuracy
        selected_contrast_images[cls] = resized_image #enhance_contrast(resized_image)

        # Extract text from the image
        ocr_result = reader.readtext(selected_contrast_images[cls], allowlist='0123456789', low_text=0.3, adjust_contrast=0.7)
        detected_texts = [filter_score(text[1]) for text in ocr_result]
        extracted_scores.append((cls, detected_texts))

    # Step 3: Determine final scores
    score1, score2 = prev_score1, prev_score2

    for cls, texts in extracted_scores:
        if cls == 0 and texts:
            score1 = texts[0]
        elif cls == 1 and texts:
            score2 = texts[0]

    prev_score1, prev_score2 = score1, score2

    # Step 4: Save timestamps for score changes
    timestep = datetime.now().strftime("%H:%M:%S")
    scores_data.append([timestep, score1, score2])

    return selected_contrast_images, score1, score2