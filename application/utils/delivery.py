import tkinter as tk
from tkinter import filedialog
import cv2
import easyocr
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import numpy as np
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

# Initialize Tkinter
root = tk.Tk()
root.title("Real-time Sports Score Detection")
root.geometry("800x600")  # Smaller default size for laptops
root.configure(bg="#2C3E50")  # Dark background

# Make the window resizable
root.resizable(True, True)

# Video Frame
video_frame = tk.Frame(root, bg="black")
video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

video_label = tk.Label(video_frame, bg="black")
video_label.pack(fill=tk.BOTH, expand=True)

# Placeholder image
placeholder_image = Image.new("RGB", (640, 480), "black")
placeholder_photo = ImageTk.PhotoImage(placeholder_image)
video_label.config(image=placeholder_photo)
video_label.image = placeholder_photo

# Score Display Frame
score_frame = tk.Frame(root, bg="#ECF0F1", padx=10, pady=10, bd=5, relief="ridge")
score_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

tk.Label(score_frame, text="Live Score", font=("Arial", 18, "bold"), bg="#ECF0F1", fg="#2C3E50").pack(pady=5)

score_container = tk.Frame(score_frame, bg="white", padx=10, pady=10, bd=3, relief="solid")
score_container.pack()

# Score Headers
header1 = tk.Label(score_container, text="Score 1", font=("Arial", 14, "bold"), fg="#E74C3C", bg="white", width=10)
header1.grid(row=0, column=0, padx=5, pady=5)
header2 = tk.Label(score_container, text="Score 2", font=("Arial", 14, "bold"), fg="#2980B9", bg="white", width=10)
header2.grid(row=0, column=1, padx=5, pady=5)

# Dynamic Score Labels
score1_label = tk.Label(score_container, text="--", font=("Arial", 20, "bold"), bg="white", fg="black", width=10)
score1_label.grid(row=1, column=0, padx=5, pady=5)
score2_label = tk.Label(score_container, text="--", font=("Arial", 20, "bold"), bg="white", fg="black", width=10)
score2_label.grid(row=1, column=1, padx=5, pady=5)

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
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.webm")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        # Reset scores
        score1_label.config(text="--")
        score2_label.config(text="--")
        # Show first frame
        ret, frame = cap.read()
        if ret:
            display_frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk)
            video_label.image = imgtk

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
    save_scores_to_csv()

def save_scores_to_csv():
    global scores_data, start_time
    if not scores_data:
        print("No scores to save.")
        return

def restart_process():
    global video_path, cap, is_running, scores_data
    is_running = False
    if cap:
        cap.release()
    video_path = ""
    cap = None
    scores_data = []  # Clear scores data
    score1_label.config(text="--")
    score2_label.config(text="--")
    video_label.config(image=placeholder_photo)
    video_label.image = placeholder_photo

def close_window():
    global is_running, cap
    is_running = False
    if cap:
        cap.release()
    root.destroy()

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