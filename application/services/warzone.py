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
import easyocr
from queue import Empty
from ..services.machine_learning import detect_text_with_api_key
from ..utils.delivery import process_video

logging.basicConfig(level=logging.INFO)
os.environ["KERAS_BACKEND"] = "jax"
reader = easyocr.Reader(['en'])

# Global frame buffer
start_frame_buffer = deque(maxlen=30)
end_frame_buffer = deque(maxlen=30)
spectating_frame_buffer = deque(maxlen=30)

# Define multipliers
TOP_5_MULTIPLIER = 1.25
TOP_3_MULTIPLIER = 1.5
VICTORY_MULTIPLIER = 2.0

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

def calc_sg_score(kill_count, ranking):
    """
    Calculate the sgScore based on the ranking and kill count.
    - Victory (Rank 1) → x2.0
    - Top 3 (Ranks 2-3) → x1.5
    - Top 5 (Ranks 4-5) → x1.25
    - Otherwise → No multiplier
    """
    # Ensure ranking and kill count are valid numbers
    ranking = float(ranking)
    kill_count = float(kill_count)

    print(f"Ranking: {ranking}, Kill Count: {kill_count}")  # Debug: print the values

    if ranking == 1:
        return kill_count * VICTORY_MULTIPLIER
    elif 2 <= ranking <= 3:  # Equivalent to ranking in [2, 3]
        return kill_count * TOP_3_MULTIPLIER
    elif 4 <= ranking <= 5:  # Equivalent to ranking in [4, 5]
        return kill_count * TOP_5_MULTIPLIER
    else:
        return kill_count  # No multiplier for ranks greater than 5

 
def update_match_count(event_id, user_id):
    """
    Increment the match_count field for the given user in the specified event.
    If match_count doesn't exist, initialize it at 0.
    """
    # Get the path to the user's match count
    path = f'event-{event_id}/{user_id}/matchCount'
    db_ref = db.reference(path)

    try:
        # Fetch the current match_count value
        current_match_count = db_ref.get()

        if current_match_count is None:
            # If the match_count doesn't exist, initialize it to 0
            current_match_count = 0

        # Increment the match_count by 1
        new_match_count = current_match_count + 1

        # Update Firebase with the new match_count
        db_ref.set(new_match_count)

        print(f"Match count for user {user_id} in event {event_id} updated to {new_match_count}")

    except Exception as e:
        logging.error(f"Error updating match count for user {user_id} in event {event_id}: {e}")
        
def get_match_count(event_id, user_id):
    """
    Fetch and increment the match count for a given user and event.
    """
    # Get the path to the user's match count
    path = f'event-{event_id}/{user_id}/matchCount'
    db_ref = db.reference(path)

    try:
        # Fetch the current match_count value
        current_match_count = db_ref.get()

        if current_match_count is None:
            # If the match_count doesn't exist, initialize it to 0
            current_match_count = 0

        # Increment the match_count by 1
        new_match_count = current_match_count + 1

        # Update Firebase with the new match_count
        db_ref.set(new_match_count)

        print(f"Match count for user {user_id} in event {event_id} updated to {new_match_count}")
        return new_match_count

    except Exception as e:
        logging.error(f"Error fetching and updating match count for user {user_id} in event {event_id}: {e}")
        return None
    
def update_firebase_match_ranking_and_score(user_id, event_id, match_count, ranking, kill_count, max_retries=3):
    """
    Calculate the SG score, update Firebase with the ranking, kill count, and SG score for a given match.
    All match data is stored within the matchHistory path for the user.
    """
    # Calculate SG score based on ranking and kill count
    sg_score = calc_sg_score(kill_count, ranking)

    # Define the Firebase path for match history
    path = f'event-{event_id}/{user_id}/matchHistory'

    db_ref = db.reference(path)
    
    for attempt in range(max_retries):
        try:
            # Fetch current match history object from Firebase
            current_data = db_ref.get()
            
            if current_data is None:
                current_data = {}  # Initialize if empty

            # Update or create a new match entry with ranking, kill count, and SG score
            current_data[f'match_{match_count}'] = {
                'ranking': float(ranking),
                'killCount': float(kill_count), 
                'sgScore': float(sg_score)  # Include SG score for each match
            }

            # Update the Firebase with the new match data
            db_ref.set(current_data)
            
            break  # Exit the loop if the operation is successful

        except Exception as e:
            logging.error(f"Error updating Firebase matchRanking and sgScore: {e}. Attempt {attempt + 1} of {max_retries}.")
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

def handle_match_state(frame, user_id, event_id):
    placement_map = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4, "5TH": 5}
    # Use the API function instead of pytesseract
    detected_text = detect_text_with_api_key(frame)
    placement_value = None  # Default value
    start_pattern = None  # Default value
    ad_pattern = None  # Default value
    
    # Flatten the list of detected_text and convert each item to uppercase to check for "PLACE"
    flattened_data = []
    for item in detected_text:  # Assuming detect_text_with_api_key returns a list of strings
        flattened_data.extend(item.split())  # Split string elements into words

    # Check if "PLACE" is in any of the strings and find the placement before it
    for i in range(len(flattened_data)):
        if flattened_data[i].upper() == "PLACE":  # Check if "PLACE" is found
            if i > 0:  # Make sure there is a word before "PLACE"
                placement = flattened_data[i - 1].upper()
                if placement in placement_map:
                    placement_value = placement_map[placement]  # Store placement number
                    print("Placement Number:", placement_value)
                    break
    
    # Check if the flattened data contains "ENTERING THE WARZONE"
    for item in flattened_data:
        if "ENTERING THE WARZONE" in item.upper():
            start_pattern = True  # Match found for "Entering the Warzone"
            break  # No need to check further if we found the pattern
        elif "COMMERCIAL" in item.upper():
            ad_pattern = True
    
    if start_pattern:
        return "start_match"
    elif placement_value:
        return "end_match"
    elif ad_pattern:
        return "ad_displaying"
    
    return "in_match"

def process_frame(frame, event_id, user_id, match_count, match_count_updated, frame_count):
        
     # Check the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]
    expected_width = 1920  # Replace with the expected width
    expected_height = 1080  # Replace with the expected height

    if frame_width != expected_width or frame_height != expected_height:
        print(f"Frame {frame_count}: Unexpected frame size: {frame_width}x{frame_height}")
        return  # Skip processing if the size is unexpected

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"/Users/trell/Projects/machine-learning/frames/debug_frame_{frame_count}.jpg", gray_frame)  
    #### REFACTOR THIS FUNCTION LATER####  
    detected_text = pytesseract.image_to_string(frame)
    spectating_pattern_found = process_frame_for_detection(detected_text, spectating_frame_buffer, ["spectating"])
    update_firebase(user_id, event_id, spectating_pattern_found)
    match_state = handle_match_state(gray_frame, user_id, event_id)
    
    if match_state == 'start_match':
        print(f"Processing start match for user {user_id} in event {event_id}...")
        if match_count_updated.value == 1:
            match_count_updated.value == 0
    elif match_state == 'ad_displaying':
        print(f"Processing ad displaying for user {user_id} in event {event_id}...")      
    elif match_state == 'in_match':
        print(f"Processing in match for user {user_id} in event {event_id}...")
        if match_count_updated.value == 1:
            match_count_updated.value == 0
        process_frame_scores(event_id, user_id, match_count.value, frame, frame_count)    
    elif match_state == 'end_match':
        print(f"Processing end match for user {user_id} in event {event_id}...")
        if match_count_updated.value == 0:
            update_match_count(event_id, user_id)
            match_count.value = get_match_count(event_id, user_id)
            match_count_updated.value == 1
    else:
        raise ValueError(f"Unknown match state: {match_state}")

def frame_worker(frame_queue, event_id, user_id, match_count, match_count_updated):
    start_time = time.time()  # Start total timer
    frame_count = 0  # Track number of frames processed

    while True:        
        try:
            frame, frame_count = frame_queue.get(timeout=5)
            if frame is None:
                logging.info("Received termination signal (None). Exiting loop.")
                break  # Stop processing

            logging.info(f"Processing frame {frame_count}")  # Debug log
            process_frame(frame, event_id, user_id, match_count, match_count_updated, frame_count)

        except Empty:
            logging.warning("Frame queue is empty. Waiting for frames...")
            continue  # Keep waiting if the queue is empty

    total_end_time = time.time()  # End total timer
    total_elapsed_time = total_end_time - start_time
    logging.info(f"Total frames processed: {frame_count}")
    logging.info(f"Total execution time of frame_worker: {total_elapsed_time:.4f} seconds")
    logging.info("Frame worker exiting")

def get_second_valid_number(detected_texts):
    if len(detected_texts) < 2:  # Ensure there are at least two elements
        return None

    second_value = detected_texts[1]  # Get the second value

    if second_value.isdigit():  # Check if it's a valid number
        return int(second_value)  # Convert to integer

    return None  # Return None if not a valid number

def get_last_valid_number(detected_texts):
    for item in reversed(detected_texts):  # Iterate from the last element to the first
        if item.isdigit():  # Check if it's a valid number
            return int(item)  # Convert to integer and return
    return None  # Return None if no valid number is found


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def process_frame_scores(event_id, user_id, match_count, frame, frame_count):
    """
    Procsesses the top-right corner of a frame, resizes it, and detects text using a given text detection function.

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
        #### TODO - Piece of code to help debug the frame cut out.
        #### This is to write the image to a folder in project.
        output_dir = f'/Users/trell/Projects/machine-learning/frames_processed'
        output_filename = f"{output_dir}/new_processed_frame_2{frame_count}.jpg"

        #### TODO - New Delivery.Py file using model lets try this
        frame_rgb, score1, score2 = process_video(frame)
        update_firebase_match_ranking_and_score(user_id, event_id, match_count, score1, score2)
        cv2.imwrite(output_filename, frame_rgb)
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
        
        match_count = manager.Value('i', 0)  # Shared integer variable
        match_count_updated = manager.Value('i', 0)  # 0 = False, 1 = True
        
        for _ in range(num_workers):
            p = Process(target=frame_worker, args=(frame_queue, event_id, user_id, match_count, match_count_updated))
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