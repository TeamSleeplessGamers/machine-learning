import cv2
import pytesseract
import time
import logging
from collections import deque
from firebase_admin import db

logging.basicConfig(level=logging.INFO)

# Define global variables if needed
detection_count = 0
frame_buffer = deque(maxlen=30)

def analyze_buffer(buffer, threshold=10):
    """
    Analyze the buffer to determine if a pattern is detected.
    """
    # Check if the buffer has consistent zeros (indicating "SPECTATING" has not been found)
    pattern_detected = False
    
    # Ensure we only check when buffer is filled to its maxlen
    if len(buffer) == buffer.maxlen:
        zero_count = buffer.count(0)
        # If the count of zeros is greater than the threshold, return False
        if zero_count >= threshold:
            pattern_detected = True
            
    return pattern_detected
    
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

        _, binary_frame = cv2.threshold(resized_frame, 128, 255, cv2.THRESH_BINARY)
        blurred_frame = cv2.GaussianBlur(binary_frame, (5, 5), 0)
        detected_text = pytesseract.image_to_string(blurred_frame)

        # Save the processed frame to the specified directory
        frame_path = f'./process_frames/frame_{frame_count}.jpg'
        cv2.imwrite(frame_path, blurred_frame)
        
        # Search for the word "SPECTATING" in the detected text (case-insensitive)
        if "spectating".lower() in detected_text.lower():
            detection_count += 1
        else:
            detection_count = 0 

        frame_buffer.append(detection_count)
        # Analyze the buffer if it has at least 10 frames
        if len(frame_buffer) >= 10:
            pattern_found = analyze_buffer(frame_buffer)
            update_firebase(user_id, event_id, not pattern_found)    
    cap.release()
    cv2.destroyAllWindows()
