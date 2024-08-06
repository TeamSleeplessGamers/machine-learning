import cv2
import pytesseract
import time
import os
import logging
from fuzzywuzzy import process
from ..config.firebase import initialize_firebase
from firebase_admin import db
from collections import deque
from multiprocessing import Process, Manager, Queue
from queue import Empty

logging.basicConfig(level=logging.INFO)

# Initialize Firebase
initialize_firebase()

# Global frame buffer
frame_buffer = deque(maxlen=30)

def analyze_buffer(buffer, threshold=5):
    non_zero_count = sum(1 for value in buffer if value > 0)
    return non_zero_count >= threshold

def update_firebase(user_id, event_id, is_spectating, max_retries=3):
    path = f'event-{event_id}/{user_id}'
    db_ref = db.reference(path)
    
    for attempt in range(max_retries):
        try:
            db_ref.update({'isSpectating': is_spectating})
            break
        except Exception as e:
            logging.error(f"Error updating Firebase: {e}. Attempt {attempt + 1} of {max_retries}.")
            time.sleep(2)

def match_text_with_known_words(text, known_words):
    matched_words = []
    words = text.split()
    
    for word in words:
        if not word.isalnum() or len(word) < 3:
            continue
        
        closest_match, score = process.extractOne(word, known_words)
        if score >= 70:
            matched_words.append(word)
    return ' '.join(matched_words)

def process_frame(frame, event_id, user_id):
    global frame_buffer

    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Add your processing logic here
    except cv2.error as e:
        logging.error(f"Error processing frame: {e}")
        
    height, width = gray_frame.shape
    new_width = int(width * 2)
    new_height = int(height * 2)
    resized_frame = cv2.resize(gray_frame, (new_width, new_height))
    frame_invert = cv2.bitwise_not(resized_frame)
    frame_scale_abs = cv2.convertScaleAbs(frame_invert, alpha=1.0, beta=0)
    custom_config = r'--oem 3 --psm 6'
    detected_text = pytesseract.image_to_string(frame_scale_abs, config=custom_config)

    detection_count = 0
    if "spectating".lower() in detected_text.lower():
        detection_count += 1
    else:
        known_words = ["SPECTATING"]
        corrected_text = match_text_with_known_words(detected_text, known_words)
        if corrected_text:
            detection_count += 1
        detection_count = 0 
    frame_buffer.append(detection_count)
    
    pattern_found = analyze_buffer(frame_buffer)
    update_firebase(user_id, event_id, pattern_found)
    return detection_count

def frame_worker(frame_queue, event_id, user_id):
    while True:
        try:
            frame, frame_count = frame_queue.get(timeout=5)
            if frame is None:
                break
            process_frame(frame, event_id, user_id)
        except Empty:
            continue
    logging.info("Frame worker exiting")

def match_template_spectating_in_video(video_path, event_id=None, user_id=None):
    with Manager() as manager:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        frame_queue = Queue(maxsize=10)
        num_workers = 4
        workers = []
        
        for _ in range(num_workers):
            p = Process(target=frame_worker, args=(frame_queue, event_id, user_id))
            p.start()
            workers.append(p)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 300 == 0:
                if not frame_queue.full():
                    frame_queue.put((frame, frame_count))
                else:
                    logging.warning("Frame queue is full, skipping frame")

        cap.release()
        cv2.destroyAllWindows()

        for _ in range(num_workers):
            frame_queue.put((None, None))  # Sentinel values to stop workers
        
        for p in workers:
            p.join()
