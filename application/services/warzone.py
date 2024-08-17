import cv2
import pytesseract
import time
import logging
import re
from fuzzywuzzy import process
from firebase_admin import db
from collections import deque
from multiprocessing import Process, Manager, Queue
from queue import Empty

logging.basicConfig(level=logging.INFO)

# Global frame buffer
start_frame_buffer = deque(maxlen=30)
spectating_frame_buffer = deque(maxlen=30)

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
    return pattern_found

def process_frame(frame, event_id, user_id, frame_count):
    rectangles = [
        (1798, 50, 60, 30),
        (25, 300, 50, 50),
    ]
    test_image_path = '/Users/trell/Projects/machine-learning/frames/start_test_frame.jpg'
    test_image = cv2.imread(test_image_path)

    if test_image is None:
        print(f"Error: Could not open or find the image at {test_image_path}")
    
    test_gray_frame = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'/Users/trell/Projects/machine-learning/frames/output_gray_frame_{frame_count}.jpg', gray_frame)
    except cv2.error as e:
        logging.error(f"Error processing frame: {e}")
      
    height, width = gray_frame.shape
    new_width = int(width * 2)
    new_height = int(height * 2)
    resized_frame = cv2.resize(gray_frame, (new_width, new_height))
    frame_invert = cv2.bitwise_not(resized_frame)
    frame_scale_abs = cv2.convertScaleAbs(frame_invert, alpha=1.0, beta=0)
    custom_config = r'--oem 3 --psm 6 -l eng'
    detected_text = pytesseract.image_to_string(frame_scale_abs, config=custom_config)

    output_dir = f'/Users/trell/Projects/machine-learning/frames_processed'
    custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789IO'

    match_state = handle_match_state(gray_frame)
    
    if match_state == 'start_match':
        print("Processing start match...")
    elif match_state == 'ad_displaying':
        print("Processing ad displaying...")        
    elif match_state == 'in_match':
        process_and_extract_text(frame, rectangles, output_dir, frame_count, custom_config)
    elif match_state == 'end_match':
        print("Processing end match...")
    else:
        raise ValueError(f"Unknown match state: {match_state}")
        
    spectating_pattern_found = process_frame_for_detection(detected_text, spectating_frame_buffer, ["spectating"])
    update_firebase(user_id, event_id, spectating_pattern_found)

def frame_worker(frame_queue, event_id, user_id):
    while True:
        try:
            frame, frame_count = frame_queue.get(timeout=5)
            if frame is None:
                break
            process_frame(frame, event_id, user_id, frame_count)
        except Empty:
            continue
    logging.info("Frame worker exiting")

def process_and_extract_text(frame, rectangles, output_dir, frame_count, custom_config=None):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    color_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

    extracted_texts = []
    for i, (x, y, w, h) in enumerate(rectangles):
        roi_frame = color_frame[y:y + h, x:x + w]

        scale_factor = 2
        rescaled_roi = cv2.resize(roi_frame, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)

        rgb_roi_frame = cv2.cvtColor(rescaled_roi, cv2.COLOR_BGR2RGB)
        output_path = f'{output_dir}/extracted_region_{i + 1}.jpg'
        cv2.imwrite(output_path, rgb_roi_frame)

        if not custom_config:
            custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789IO'
        extracted_text = pytesseract.image_to_string(rgb_roi_frame, config=custom_config).strip()
        extracted_texts.append(extracted_text)
        print(f"Extracted text from region {i + 1}: {extracted_text}")

    for (x, y, w, h) in rectangles:
        img_with_box = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    processed_image_path = f'{output_dir}/processed_gray_frame_{frame_count}.jpg'
    cv2.imwrite(processed_image_path, img_with_box)
    
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
