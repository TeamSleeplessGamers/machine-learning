import cv2
import pytesseract
import time
import logging
from collections import deque
from firebase_admin import db
from fuzzywuzzy import fuzz, process

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
            logging.info(f"Firebase updated successfully for event {event_id}, user {user_id}.")
            break  # Exit loop if successful
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

def match_template_spectating_in_video(video_path, event_id=None, user_id=None):
    global detection_count
    global frame_buffer
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    frame_count = 0

    # Process the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 24 != 0:
            continue

        if frame.ndim == 3: 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 4: 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray_frame = frame

        height, width = gray_frame.shape
        new_width = int(width * 2)
        new_height = int(height * 2)
        resized_frame = cv2.resize(gray_frame, (new_width, new_height))
        frame_invert = cv2.bitwise_not(resized_frame)
        frame_scale_abs = cv2.convertScaleAbs(frame_invert, alpha=1.0, beta=0)
        custom_config = r'--oem 3 --psm 6'
        detected_text = pytesseract.image_to_string(frame_scale_abs, config=custom_config)

        # Search for the word "SPECTATING" in the detected text (case-insensitive)
        if "spectating".lower() in detected_text.lower():
            detection_count += 1
        else:
            # Split the detected text into words
            known_words = ["SPECTATING"]
            corrected_text = match_text_with_known_words(detected_text, known_words)
            if corrected_text:
                detection_count += 1
            detection_count = 0 

        frame_buffer.append(detection_count)

        # Analyze the buffer if it has at least 10 frames
        if len(frame_buffer) >= 10:
            pattern_found = analyze_buffer(frame_buffer)
            update_firebase(user_id, event_id, pattern_found)    
    cap.release()
    cv2.destroyAllWindows()
