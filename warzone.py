import cv2
import pytesseract
import time
import os
import logging
from collections import deque
from firebase_admin import db
from fuzzywuzzy import process
from multiprocessing import Process, Manager, Queue
from queue import Empty

logging.basicConfig(level=logging.INFO)

# Define global variables if needed
detection_count = 0
frame_buffer = deque(maxlen=30)

def analyze_buffer(buffer, threshold=10):
    """
    Analyze the buffer to determine if a pattern is detected.
    """
    # Ensure we only check when the buffer is filled to its maxlen
    if len(buffer) == buffer.maxlen:
        non_zero_count = sum(1 for value in buffer if value > 0)
        # If the count of values greater than 0 is greater than or equal to the threshold, return True
        if non_zero_count >= threshold:
            return True
    # Return False if the condition is not met
    return False
    
def update_firebase(user_id, event_id, is_spectating, max_retries=3):
    path = f'event-{event_id}/{user_id}'
    db_ref = db.reference(path)
    
    for attempt in range(max_retries):
        try:
            # Update Firebase
            db_ref.update({
                'isSpectating': is_spectating,
            })
            break
        except Exception as e:
            logging.error(f"Error updating Firebase: {e}. Attempt {attempt + 1} of {max_retries}.")
            time.sleep(2)  # Wait before retrying

def match_text_with_known_words(text, known_words):
    """
    Find the closest match for the given text from a list of known words using fuzzy matching.
    """
    matched_words = []
    words = text.split()
    
    for word in words:
        # Skip words that are too short or non-alphanumeric
        if not word.isalnum() or len(word) < 3:
            continue
        
        closest_match, score = process.extractOne(word, known_words)
        if score >= 70:  # Adjust the threshold as needed
            matched_words.append(word)
    return ' '.join(matched_words)

def save_frame(frame, frame_count, output_folder):
    """
    Save the frame as an image in the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    frame_filename = os.path.join(output_folder, f'frame_{frame_count}.png')
    cv2.imwrite(frame_filename, frame)

def process_frame(frame, frame_count, output_folder):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    new_width = int(width * 2)
    new_height = int(height * 2)
    resized_frame = cv2.resize(gray_frame, (new_width, new_height))
    frame_invert = cv2.bitwise_not(resized_frame)
    frame_scale_abs = cv2.convertScaleAbs(frame_invert, alpha=1.0, beta=0)
    custom_config = r'--oem 3 --psm 6'
    detected_text = pytesseract.image_to_string(frame_scale_abs, config=custom_config)

    # Determine if "spectating" is found in detected text
    detection_count = 0
    if "spectating".lower() in detected_text.lower():
        detection_count += 1
    else:
        known_words = ["SPECTATING"]
        corrected_text = match_text_with_known_words(detected_text, known_words)
        if corrected_text:
            detection_count += 1
        detection_count = 0 

    # Save the frame to the output folder
    save_frame(frame, frame_count, output_folder)
    print("Frame processed with detection count:", detection_count)
    return detection_count


def frame_worker(frame_queue, output_folder):
    while True:
        try:
            frame, frame_count = frame_queue.get(timeout=5)  # Timeout to prevent indefinite blocking
            if frame is None:  # Sentinel value to exit
                break
            process_frame(frame, frame_count, output_folder)
        except Empty:
            continue
    logging.info("Frame worker exiting")


def match_template_spectating_in_video(video_path, event_id=None, user_id=None):
    if not os.path.exists("processed_frames"):
        os.makedirs("processed_frames")

    # Use multiprocessing.Manager for shared state
    with Manager() as manager:
        frame_buffer = manager.list()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        frame_queue = Queue(maxsize=10)  # Queue to hold frames
        num_workers = 4
        workers = []
        
        # Start worker processes
        for _ in range(num_workers):
            p = Process(target=frame_worker, args=(frame_queue, "processed_frames"))
            p.start()
            workers.append(p)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                if not frame_queue.full():
                    frame_queue.put((frame, frame_count))
                else:
                    logging.warning("Frame queue is full, skipping frame")

        cap.release()
        cv2.destroyAllWindows()

        # Add sentinel values to shut down workers
        for _ in range(num_workers):
            frame_queue.put((None, None))  # Sentinel value
        
        # Collect results if needed
        for p in workers:
            p.join()

        # Analyze the buffer and update Firebase
        if len(frame_buffer) >= 10:
            pattern_found = analyze_buffer(frame_buffer)
            update_firebase(user_id, event_id, pattern_found)

