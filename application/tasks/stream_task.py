from flask import Blueprint, request, jsonify, send_file, Response
import pandas as pd
import time
from ..services.twitch_recorder import TwitchRecorder
from ..services.twitch_oauth import get_twitch_oauth_token
import pytesseract
import os
from firebase_admin import db
import cv2
import hashlib
import pytz
import hmac
import yaml
import logging
from ..utils.heatmap_generator import generate_heatmap
import requests
import csv
from datetime import datetime, timedelta
from .. import celery_config
import streamlink
from ultralytics import YOLO
import threading
from ..services.scheduler import start_scheduler
from datetime import datetime
from ..config.database import Database
from ..config.firebase import initialize_firebase
from ..utils.delivery import process_video, number_detector_2
from ..utils.utils import calc_sg_score

# Celery task to process the Twitch stream
@celery_config.celery.task(bind=True, max_retries=5, default_retry_delay=10, queue="gpu_tasks")
def process_twitch_stream(self, username, user_id, event_id, match_duration):
    # heavy GPU processing here
    try:
        initialize_firebase()
        spectating_state_map = {}

        # Local state variables (no multiprocessing needed)
        match_count = 0
        match_count_updated = 1
        end_match_start_time = 0.0
        flag = False

        # Stream duration: 4 hours (or from config if available)
        end_time = datetime.now() + timedelta(minutes=match_duration)

        # Twitch stream URL
        streams = streamlink.streams(f"https://www.twitch.tv/{username}")
        if not streams:
            raise Exception("Could not access Twitch stream")
        stream_url = streams["best"].url

        # Open the stream using OpenCV
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise Exception("Could not open Twitch stream")

        frame_count = 0

        while datetime.now() < end_time:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Only process every 90th frame (assuming ~30 FPS ≈ 3 seconds)
            if frame_count % 90 != 0:
                continue

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
                print(f"This {frame_count} is result for {cls}: {results}")
                filename = f"frames_processed/frame_{frame_count}.jpg"
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
    filename = f"frames/frame.jpg"
    #cv2.imwrite(filename, frame)
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


def update_firebase_match_ranking(user_id, event_id, match_count, ranking):
    """
    Update the ranking for a given match in Firebase without overwriting other fields.
    """
    # Define the Firebase path for match history, including match count and ranking
    path = f'event-{event_id}/{user_id}/matchHistory/match_{match_count}/ranking'
    db_ref = db.reference(path)

    for attempt in range(3):  # Handling retries internally
        try:
            # Prepare match data to update ranking
            if ranking is not None:
                db_ref.set(float(ranking))  # Set the ranking at the specific path

            break  # Exit the loop if the operation is successful

        except Exception as e:
            logging.error(f"Error updating Firebase match ranking: {e}. Attempt {attempt + 1}.")
            time.sleep(2)  # Wait before retrying
            
def update_firebase_match_kill_count(user_id, event_id, match_count, kill_count):
    """
    Update the kill count for a given match in Firebase without overwriting other fields.
    """
    # Define the Firebase path for match history, including match count and kill count
    path = f'event-{event_id}/{user_id}/matchHistory/match_{match_count}/killCount'
    db_ref = db.reference(path)

    for attempt in range(3):  # Handling retries internally
        try:
            # Prepare match data to update kill count
            if kill_count is not None:
                db_ref.set(float(kill_count))  # Set the kill count at the specific path

            break  # Exit the loop if the operation is successful

        except Exception as e:
            logging.error(f"Error updating Firebase match kill count: {e}. Attempt {attempt + 1}.")
            time.sleep(2)  # Wait before retrying

def update_firebase_match_ranking_and_score(user_id, event_id, match_count, ranking, kill_count):
    """
    Calculate the SG score and update Firebase with the ranking, kill count, and SG score for a given match.
    Calls separate functions to update ranking and kill count individually.
    """
    # Calculate SG score based on ranking and kill count, if both are available
    sg_score = calc_sg_score(kill_count, ranking) if ranking is not None and kill_count is not None else None

    # Update ranking if available
    if ranking is not None:
        update_firebase_match_ranking(user_id, event_id, match_count, ranking)

    # Update kill count if available
    if kill_count is not None:
        update_firebase_match_kill_count(user_id, event_id, match_count, kill_count)

    # Define the Firebase path for match history, including match count and sgScore
    path = f'event-{event_id}/{user_id}/matchHistory/match_{match_count}/sgScore'
    db_ref = db.reference(path)

    for attempt in range(3):  # Handling retries internally
        try:
            # Prepare match data to update sgScore
            if sg_score is not None:
                db_ref.set(float(sg_score))  # Set the sgScore at the specific path

            break  # Exit the loop if the operation is successful

        except Exception as e:
            logging.error(f"Error updating Firebase match sgScore: {e}. Attempt {attempt + 1}.")
            time.sleep(2)  # Wait before retrying
                
 
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
 
