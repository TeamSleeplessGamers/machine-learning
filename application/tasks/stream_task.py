import time
from ..services.twitch_recorder import TwitchRecorder
from ..services.twitch_oauth import get_twitch_oauth_token
from firebase_admin import db
import cv2
import os
import logging
from ..utils.heatmap_generator import generate_heatmap
from datetime import datetime, timedelta
from .. import celery_config
import streamlink
import numpy as np
import ffmpeg
from datetime import datetime
from ..config.firebase import initialize_firebase
from ..utils.delivery import process_video, number_detector_2
from ..utils.utils import calc_sg_score

initialize_firebase()

# Celery task to process the Twitch stream
@celery_config.celery.task(bind=True, max_retries=5, default_retry_delay=10)
def process_twitch_stream(self, username, user_id, event_id, match_duration):
    # heavy GPU processing here
    try:
        spectating_state_map = {}

        # Local state variables (no multiprocessing needed)
        match_count = 0
        match_count_updated = 1
        end_match_start_time = 0.0
        flag = False

        end_time = datetime.now() + timedelta(minutes=match_duration if match_duration else 60)
        streams = streamlink.streams(f"https://www.twitch.tv/{username}")
        stream_url = streams["best"].url

        if not stream_url:
            print(f"No 'best' stream quality found for {username}")
            return
        
        cap, process = get_stream_capture(stream_url)

        # If using ffmpeg, get resolution
        if process:
            width, height = get_stream_resolution(stream_url)
        else:
            print(f"Error probing stream resolution: {e}")
            width = height = None
    
        frame_count = 0

        while datetime.now() < end_time:
            ret, frame = read_frame(cap, process, width, height)
            if not ret or frame is None:
                continue

            frame_count += 1
            if frame_count % 90 != 0:
                continue

            #filename = f"frames_processed/frame_{frame_count}.jpg"
            #cv2.imwrite(filename, frame)
    
            match_state = handle_match_state(frame)
            spectating_pattern_found = match_state == "spectating"
            state_key = (user_id, event_id)
            last_state = spectating_state_map.get(state_key)

            if last_state != spectating_pattern_found:
                update_firebase(user_id, event_id, spectating_pattern_found)
                spectating_state_map[state_key] = spectating_pattern_found

            if spectating_pattern_found:
                return

            if end_match_start_time != 0.0:
                elapsed_time = time.time() - end_match_start_time
                if elapsed_time > 30:
                    flag = False
                    end_match_start_time = 0.0

            if match_state == 'start_match':
                if match_count_updated == 1:
                    match_count_updated = 0

            elif isinstance(match_state, tuple) and match_state[0] == 'in_match' and end_match_start_time == 0.0:
                if match_count_updated == 1:
                    match_count_updated = 0
                _, detected_regions = match_state
                process_frame_scores(event_id, user_id, match_count, frame, frame_count, detected_regions)

            elif match_state == 'end_match' and not flag:
                if match_count_updated == 0:
                    update_match_count(event_id, user_id, match_count)
                    match_count_updated = 1
                    end_match_start_time = time.time()
                flag = True
        cap.release()
    except Exception as e:
        logging.error(f"Error processing Twitch stream: {e}")
        raise self.retry(exc=e)

def get_stream_capture(stream_url):
    if os.environ.get("ENV") == "production":
        process = (
            ffmpeg
            .input(stream_url, hwaccel='cuda')  # use 'videotoolbox' on macOS, 'cuda' on Linux
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdout=True)
        )
        return None, process

    else:
        # For development, fallback to CPU decoding or use videotoolbox if on macOS
        process = (
            ffmpeg
            .input(stream_url, hwaccel='videotoolbox')  # change as needed
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdout=True)
        )
        return None, process

def get_stream_resolution(stream_url):
    probe = ffmpeg.probe(stream_url)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return int(video_stream['width']), int(video_stream['height'])

def read_frame(cap, process, width=1280, height=720):
    if cap:
        ret, frame = cap.read()
        return ret, frame
    elif process:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            return False, None
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        return True, frame
    else:
        return False, None
    
def process_frame_scores(event_id, user_id, match_count, frame, frame_count, detected_regions):
    """
    Processes the top-right corner of a frame, resizes it, and detects text using a given text detection function.

    Args:
        frame (numpy.ndarray): The input video frame.
        frame_count (int): The frame number (used for logging).
        event_id (int): The event ID.
        user_id (int): The user ID.
        match_count (int): The match count.
        
    Returns:
        None
    """
    try:        
        # Initialize a dictionary to store the combined results
        combined_results = {}
        
        # Step 2: Process each detected class and image
        for cls, image in detected_regions.items():
            if image is not None and image.size > 0:
                results = number_detector_2(image) # detect_text_with_api_key(image)
                #cv2.imwrite(filename, image)
                # Check if detected_text is a valid number
                if results is None:
                    results = None  # Set to None if not a valid number
                
                # Add the result to the combined_results dictionary
                combined_results[cls] = results
                # Debugging: Save the image
                #output_filename = f"/Users/trell/Projects/machine-learning-2/frames_processed/processed_frame_{frame_count}_class_{cls}.jpg"
                #cv2.imwrite(output_filename, image)  
            else:
                print(f"No valid detection for Class {cls}")
        # Step 3: Extract ranking from the combined results (using cls 0 as the ranking)
        ranking = combined_results.get('team_ranking', None)
        kills_count = combined_results.get('user_kills', None)
        
        update_firebase_match_ranking_and_score(
            user_id, event_id, match_count, ranking, kills_count
        )
        
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")

def handle_match_state(frame):
    detected_regions = process_video(frame)

    if 'user_deploying' in detected_regions:
        return "start_match"
    elif 'final_kill_cam' in detected_regions or 'warzone_victory' in detected_regions or 'warzone_defeat' in detected_regions:
        return "end_match"
    elif 'user_spectating' in detected_regions:
        return "spectating"
    elif 'team_ranking' in detected_regions or 'match_users_left' in detected_regions or 'user_kills' in detected_regions:
        in_match_regions = {k: v for k, v in detected_regions.items() if k in {'team_ranking', 'match_users_left', 'user_kills'}}
        return ("in_match", in_match_regions)
    else:
        return "ad_displaying"
    
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

def update_match_count(event_id, user_id, match_count):
    """
    Increment the match_count field for the given user in the specified event.
    If match_count doesn't exist, initialize it at 0. The increment is done in multiples of 5.
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

        new_match_count = current_match_count + 1
        match_count.value = new_match_count

        # Update Firebase with the new match_count
        db_ref.set(new_match_count)
    except Exception as e:
        logging.error(f"Error updating match count for user {str(user_id)} in event {str(event_id)}: {e}")

def update_firebase_match_ranking_and_score(user_id, event_id, match_count, ranking, kill_count):
    """
    Calculate the SG score and update Firebase with the ranking, kill count, and SG score in a single update.
    """
    # Calculate SG score
    sg_score = calc_sg_score(kill_count, ranking) if ranking is not None and kill_count is not None else None

    # Define the Firebase path to the match data
    path = f'event-{event_id}/{user_id}/matchHistory/match_{match_count}'
    db_ref = db.reference(path)

    update_data = {}

    if ranking is not None:
        update_data["ranking"] = float(ranking)

    if kill_count is not None:
        update_data["killCount"] = float(kill_count)

    if sg_score is not None:
        update_data["sgScore"] = float(sg_score)

    for attempt in range(3):
        try:
            if update_data:
                db_ref.update(update_data)
            break
        except Exception as e:
            logging.error(f"Error updating Firebase match data: {e}. Attempt {attempt + 1}.")
            time.sleep(2)
            
def init_data(event_id, user_id, team_id=None, max_retries=3):
    """
    Initialize match_0 data in Firebase for a given event and user with default values.
    
    Args:
        event_id (str): The event ID.
        user_id (str): The user ID.
        team_id (str, optional): The team ID to store at the user level (not in matchHistory).
        max_retries (int): The maximum number of retries if the operation fails.
    """
    # Firebase path to the user's event data
    user_path = f'event-{event_id}/{user_id}'
    db_ref = db.reference(user_path)

    for attempt in range(max_retries):
        try:
            # Fetch current user-level data
            current_data = db_ref.get() or {}

            # Only set teamId once at the top level (if provided and not already present)
            if team_id is not None and 'teamId' not in current_data:
                current_data['teamId'] = team_id

            # Set default match_0 data under matchHistory
            if 'matchHistory' not in current_data:
                current_data['matchHistory'] = {}

            current_data['matchHistory']['match_0'] = {
                'ranking': 0,
                'killCount': 0,
                'sgScore': 0
            }

            # Write updated data back to Firebase
            db_ref.set(current_data)

            print(f"Initialized match_0 data for {user_id} in event {event_id}")
            break

        except Exception as e:
            logging.error(f"Error initializing match data: {e}. Attempt {attempt + 1} of {max_retries}.")
            time.sleep(2)
 
