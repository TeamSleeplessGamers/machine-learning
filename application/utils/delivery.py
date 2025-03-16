import tkinter as tk
from tkinter import filedialog
import cv2
import easyocr
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import csv
from datetime import datetime

# Load YOLO model
model = YOLO('model/score-detector.pt')
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

def upload_video():
    global video_path, cap
    video_path = filedialog.askopenfilename()
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

def start_inference():
    global is_running, scores_data, start_time
    if not video_path:
        print("Error: No video uploaded.")
        return
    if cap is None or not cap.isOpened():
        print("Error: Video not loaded.")
        return

    is_running = True
    scores_data = []  # Reset scores data
    start_time = datetime.now()  # Record the start time
    process_video()

def stop_inference():
    global is_running
    is_running = False
    # save_scores_to_csv()

def save_scores_to_csv():
    global scores_data, start_time
    if not scores_data:
        print("No scores to save.")
        return
    # Generate a unique filename using the start time
    filename = start_time.strftime("scores_data/scores_%Y%m%d_%H%M%S.csv")
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "Score 1", "Score 2"])  # Write header
        writer.writerows(scores_data)  # Write scores data
    print(f"Scores saved to {filename}")

def restart_process():
    global video_path, cap, is_running, scores_data
    is_running = False
    if cap:
        cap.release()
    video_path = ""
    cap = None
    scores_data = []  # Clear scores data

def close_window():
    global is_running, cap
    is_running = False
    if cap:
        cap.release()

def process_video(frame):
    global is_running, cap, scores_data
    prev_score1, prev_score2 = "0", "0"

    results = model(frame, verbose=False)
    extracted_scores = []

    # Dictionary to store the highest confidence detection for each class
    class_detections = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)  # Class ID
            conf = float(box.conf)  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep only the highest confidence detection for each class
            if cls not in class_detections or conf > class_detections[cls]['conf']:
                class_detections[cls] = {'conf': conf, 'box': (x1, y1, x2, y2)}

    # Process the highest confidence detections
    for cls, detection in class_detections.items():
        x1, y1, x2, y2 = detection['box']
        detected_region = frame[y1:y2, x1:x2]
        height_roi = detected_region.shape[0]

        bottom_half = detected_region[int(height_roi // 2.5): height_roi, :]
        resized_image = cv2.resize(bottom_half, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        ocr_result = reader.readtext(resized_image)

        detected_texts = [filter_score(text[1]) for text in ocr_result]
        extracted_scores.append((cls, detected_texts))

    # Initialize scores for both classes
    score1 = prev_score1
    score2 = prev_score2

    # Map detected scores to their respective classes
    for cls, texts in extracted_scores:
        if cls == 0 and len(texts) > 0:  # Class 0 corresponds to score1
            score1 = texts[0]
        elif cls == 1 and len(texts) > 0:  # Class 1 corresponds to score2
            score2 = texts[0]

    # Update previous scores
    prev_score1, prev_score2 = score1, score2

    # Draw bounding boxes with different colors for each class
    for cls, detection in class_detections.items():
        x1, y1, x2, y2 = detection['box']
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for class 0, Red for class 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    display_frame = cv2.resize(frame, (640, 480))

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    return frame_rgb, score1, score2  # Returning processed frame and scores