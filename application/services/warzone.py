import cv2
import pytesseract
import time
import logging
import os
import re
from fuzzywuzzy import process
from firebase_admin import db
import tensorflow as tf
import numpy as np
from collections import deque
from multiprocessing import Process, Manager, Queue
from queue import Empty
from ..services.machine_learning import detect_text_with_api_key

logging.basicConfig(level=logging.INFO)
os.environ["KERAS_BACKEND"] = "jax"

# Global frame buffer
start_frame_buffer = deque(maxlen=30)
spectating_frame_buffer = deque(maxlen=30)

def analyze_buffer(buffer, threshold=5):
    if not buffer:
        return False
    non_zero_count = sum(1 for value in buffer if value > 0)
    return non_zero_count > threshold

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

def process_frame_for_detection(detected_text, frame_buffer, known_words, threshold=5):
    detection_count = 0

    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, known_words)) + r')\b', re.IGNORECASE)
    if pattern.search(detected_text):
        detection_count = 1
    else:
        corrected_text = match_text_with_known_words(detected_text, known_words)
        if corrected_text:
            detection_count = 1
        else:
            detection_count = 0
          
    frame_buffer.append(detection_count)
    pattern_found = analyze_buffer(frame_buffer, threshold)
    return pattern_found

def handle_match_state(frame):
    detected_text = pytesseract.image_to_string(frame)
    pattern_found = process_frame_for_detection(detected_text, start_frame_buffer, ["Entering the Warzone"], 0)
    
    if pattern_found:
        return "start_match"
    return "in_match"

def process_frame(frame, event_id, user_id, frame_count):
     # Check the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]
    expected_width = 1920  # Replace with the expected width
    expected_height = 1080  # Replace with the expected height

    if frame_width != expected_width or frame_height != expected_height:
        print(f"Frame {frame_count}: Unexpected frame size: {frame_width}x{frame_height}")
        return  # Skip processing if the size is unexpected

    top_left = (1712, 100)
    bottom_right = (1760, 150)
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    roi_resized = cv2.resize(roi, (28, 28))  # Resize to match CNN input
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    roi_normalized = roi_gray / 255.0  # Normalize pixel values to [0, 1]
    roi_normalized = roi_normalized.reshape(1, 28, 28, 1)  # Reshape for the CNN
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width = 1800
    height = 700
    corner_size = 300
    _, original_width = gray_frame.shape
    top_right_corner = gray_frame[
        0:corner_size,
        max(0, original_width - corner_size):original_width
    ]
    resized_corner = cv2.resize(top_right_corner, (width, height))

    if len(resized_corner.shape) == 3 and resized_corner.shape[2] == 3:
        resized_corner = cv2.cvtColor(resized_corner, cv2.COLOR_BGR2GRAY)
    _, thresh_2 = cv2.threshold(resized_corner,127,255, cv2.THRESH_TOZERO)
            
    frame_invert = cv2.bitwise_not(gray_frame)
    frame_scale_abs = cv2.convertScaleAbs(frame_invert, alpha=1.0, beta=0)
    custom_config = r'--oem 3 --psm 6 -l eng'
    detected_text = pytesseract.image_to_string(frame_scale_abs, config=custom_config)

    output_dir = f'/Users/trell/Projects/machine-learning/frames_processed'
    custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789IO'

    _, thresh = cv2.threshold(gray_frame,128, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S, ltype=cv2.CV_32S)

    min_area = 10 
    max_area = 2000
    second_min_area = 50
    second_max_area = 140
    target_warzone_circle_centroid = (151, 371)
    target_warzone_kill_centroid = (1840, 115)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # CutOut For Warzone2 Resurgance circle closing
        if min_area <= area <= max_area:
            distance_to_target = np.sqrt((cx - target_warzone_circle_centroid[0])**2 + (cy - target_warzone_circle_centroid[1])**2)
            if distance_to_target < 10:    
                roi = gray_frame[y:y+h, x:x+w]        
                _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi_resized = cv2.resize(roi_bin, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                custom_config = r'--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
                extracted_text = pytesseract.image_to_string(roi_resized, config=custom_config).strip()  
        # CutOut For Warzone2 Kills
        if second_min_area <= area <= second_max_area:
            distance_to_target_2 = np.sqrt((cx - target_warzone_kill_centroid[0])**2 + (cy - target_warzone_kill_centroid[1])**2)
            if distance_to_target_2 < 100: 
                print("object", i, "what is distance", distance_to_target_2) 
                padding = 10
                padded_x = x - padding
                padded_y = y - padding
                padded_w = w + 2 * padding
                padded_h = h + 2 * padding
                
                padded_x = max(padded_x, 0)
                padded_y = max(padded_y, 0)
                padded_w = min(padded_x + padded_w, gray_frame.shape[1]) - padded_x
                padded_h = min(padded_y + padded_h, gray_frame.shape[0]) - padded_y    
    match_state = handle_match_state(gray_frame)
    
    if match_state == 'start_match':
        print("Processing start match...")
    elif match_state == 'ad_displaying':
        print("Processing ad displaying...")        
    elif match_state == 'in_match':
        print("Processing in match...")
        process_top_right_corner(frame, frame_count)    
    elif match_state == 'end_match':
        print("Processing end match...")
    else:
        raise ValueError(f"Unknown match state: {match_state}")
      
    #### REFACTOR THIS FUNCTION LATER####  
    # spectating_pattern_found = process_frame_for_detection(detected_text, spectating_frame_buffer, ["spectating"])
    # update_firebase(user_id, event_id, spectating_pattern_found)

def frame_worker(frame_queue, event_id, user_id):
    while True:
        try:
            frame, frame_count = frame_queue.get(timeout=5)
            if frame is None:
                break
            process_frame(frame, event_id, user_id, frame_count)
        except Empty:
            continue
        finally:
            logging.info("Frame worker exiting")
    logging.info("Frame worker exiting")

def process_top_right_corner(frame, frame_count, 
                             start_y=0, start_x=0, corner_size=200,
                             width=1800, height=400):
    """
    Processes the top-right corner of a frame, resizes it, and detects text using a given text detection function.

    Args:
        frame (numpy.ndarray): The input video frame.
        frame_count (int): The frame number (used for logging).
        detect_text_function (function): A function to detect text from a region.
        start_y (int): Vertical position (top offset) for the region of interest.
        start_x (int): Horizontal position (right offset) for the region of interest.
        corner_size (int): Size of the square region to extract.
        width (int): Width to resize the region.
        height (int): Height to resize the region.

    Returns:
        list: Detected texts from the region of interest.
    """
    try:
        height, width, _ = frame.shape  # Get frame dimensions
        corner_size = min(height, width) // 16  # Set corner size as a fraction of the smaller dimension

        start_y = 0  # Start from the top
        start_x = width - corner_size  # Start from the right

        top_right_corner = frame[start_y:start_y + corner_size, start_x:width]

        # Resize the selected corner region
        resized_corner = cv2.resize(top_right_corner, (width, height))

        # Call Vision API with ROI
        # detected_texts = detect_text_with_api_key(resized_corner)
        # print(f"Detected Texts (Frame {frame_count}): {detected_texts}")
        output_dir = f'/Users/trell/Projects/machine-learning/frames_processed'
        output_filename = f"{output_dir}/new_processed_frame_{frame_count}.jpg"
        cv2.imwrite(output_filename, resized_corner)

        return ""

    except Exception as e:
        print(f"Error detecting text in frame {frame_count}: {e}")
        return []
    
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
            if frame_count % 100 == 0:
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