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

database = Database()
conn = database.get_connection()

threshold = 10 
routes_bp = Blueprint('routes', __name__)

# Load game config (with error handling)
try:
    with open('../../game-config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: game-config.yaml not found. Using default configuration.")
    config = {}  # Default empty config if file is missing
except yaml.YAMLError as e:
    print(f"Warning: Error parsing game-config.yaml: {e}. Using default configuration.")
    config = {}  # Default empty config if YAML is invalid


# Twitch API Credentials
TWITCH_CLIENT_ID = "wslbro47rnh8bo8y3tl2anw0i86vbe"
TWITCH_CLIENT_SECRET = "0irkrym9153ewkcvxjrivfzlht0hxk"
TWITCH_OAUTH_URL = "https://id.twitch.tv/oauth2/token"

# Load YOLO model with dynamic path
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'model', 'score-detector-3.pt')
model = YOLO(model_path)

def get_twitch_access_token():
    payload = {
        'client_id': TWITCH_CLIENT_ID,
        'client_secret': TWITCH_CLIENT_SECRET,
        'grant_type': 'client_credentials'
    }
    response = requests.post(TWITCH_OAUTH_URL, data=payload)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Failed to get Twitch Access Token: {response.text}")


@routes_bp.route('/')
def index():
    return '''
        <h1>Welcome to Sleepless Gamers Integration</h1>
    '''

def match_template_in_video(video_path, template_path, output_folder, threshold=0.1, save_as_images=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = int(time.time())
    output_video_path = os.path.join(output_folder, f'output_video_{timestamp}.mp4')

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Error: Cannot load template image.")
        return

    template_height, template_width = template.shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if not save_as_images:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Cannot open video writer.")
            cap.release()
            return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}")

        height = 200
        width = int((height / frame.shape[0]) * frame.shape[1])
        x_start = max(frame.shape[1] - width, 0) 
        y_start = 0
        y_end = min(y_start + height, frame.shape[0])

        cropped_frame = frame[y_start:y_end, x_start:frame.shape[1] ]

        new_height = min(cropped_frame.shape[0] + 500, frame.shape[0])
        new_width = min(cropped_frame.shape[1] + 500, frame.shape[1])

        resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

        gray_cropped_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3: 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 4:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray_frame = frame

        if template.ndim == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        elif template.ndim == 4:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
        else:
            gray_template = template

        if gray_frame.dtype != gray_template.dtype:
            gray_template = gray_template.astype(gray_frame.dtype)

        if gray_frame.dtype != 'uint8' and gray_frame.dtype != 'float32':
            gray_frame = gray_frame.astype('uint8')
        if gray_template.dtype != 'uint8' and gray_template.dtype != 'float32':
            gray_template = gray_template.astype('uint8')


        detected_text = pytesseract.image_to_string(gray_frame)

        search_word = "SPECTATING".lower()
        if search_word in detected_text.lower():
            print(f"Word '{search_word.upper()}' found in the detected text.")
        else:
            print(f"Word '{search_word.upper()}' not found in the detected text.", detected_text)
            
        test_result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(test_result)

        if (template_height > gray_cropped_frame.shape[0]) or (template_width > gray_cropped_frame.shape[1]):
            print("Warning: Template is larger than the cropped frame. Skipping this frame.")
            continue

        result = cv2.matchTemplate(gray_cropped_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        reduced_width = int(template_width * 0.7)
        reduced_height = int(template_height * 1.5)
        
        top_left = None
        bottom_right = None

        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + reduced_width, top_left[1] + reduced_height)
            
            if top_left is not None and bottom_right is not None:
                cv2.rectangle(gray_cropped_frame, top_left, bottom_right, (0, 255, 0), 2)

                test_roi = cropped_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                roi = gray_cropped_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                #if roi.shape[0] > 50:
                #    roi = roi[50:, :]

                print(frame_count, f"Detected text:")
            if save_as_images:
                output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_filename, gray_frame)
            else:
                out.write(gray_cropped_frame)
    cap.release()
    if not save_as_images:
        out.release()
    cv2.destroyAllWindows()

@routes_bp.route('/match_template', methods=['POST'])
def match_template_route():
    output_folder = "./test_video"

    match_template_in_video("./processed/test_1.mp4", "./game_templates/warzone/spectating_1.jpg", output_folder)

    return jsonify({'status': 'success', 'message': 'Processing completed.'})

def check_user_online(user_login):
    twitch_client_id = os.environ['CLIENT_ID']
    
    headers = {
        'Client-ID': twitch_client_id,    
        'Authorization': f'Bearer {get_twitch_oauth_token()}'
    }
    
    stream_info_url = f'https://api.twitch.tv/helix/streams?user_login={user_login}'
    
    response = requests.get(stream_info_url, headers=headers)
    stream_data = response.json()
    
    if 'data' in stream_data and len(stream_data['data']) > 0:
        stream_info = stream_data['data'][0]
        return {
            'status': 'online',
            'stream_info': {
                'title': stream_info['title'],
                'game_name': stream_info['game_name'],
                'viewer_count': stream_info['viewer_count']
            }
        }
    else:
        return {'status': 'offline'}

def get_day_and_time():
    est = pytz.timezone('America/New_York')
    
    now_utc = datetime.now(pytz.utc)
    now_est = now_utc.astimezone(est)
    
    day_of_week = now_est.strftime("%A")
    current_hour = now_est.hour
    
    return day_of_week, current_hour

def append_to_csv(display_name, day, time):
    base_path = os.path.dirname(__file__)  
    csv_path = os.path.join(base_path, '..', '..', 'warzone-streamer.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([display_name, day, time])

@routes_bp.route('/webhooks/callback', methods=['POST'])
def webhook_callback():
    twitch_client_id = os.environ['CLIENT_ID']
    twitch_client_secret = os.environ['CLIENT_SECRET']
    headers = request.headers
    body = request.get_data(as_text=True)
    MESSAGE_TYPE_VERIFICATION = 'webhook_callback_verification'

    signature = headers.get('Twitch-Eventsub-Message-Signature')
    timestamp = headers.get('Twitch-Eventsub-Message-Timestamp')
    event_type = headers.get('Twitch-Eventsub-Message-Type')

    if signature and timestamp:
        message = headers.get('Twitch-Eventsub-Message-Id') + timestamp + body
        expected_signature = 'sha256=' + hmac.new(
            twitch_client_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, signature):
            return 'Signature verification failed', 403
        if event_type == MESSAGE_TYPE_VERIFICATION:
            return Response(request.json['challenge'], content_type='text/plain', status=200)
        if event_type == "notification":
            broadcaster_id = request.json['subscription']['condition']['broadcaster_user_id']
            subscription_type = request.json['subscription']['type']
            if len(broadcaster_id) > 0:
                api_endpoint = f"https://api.twitch.tv/helix/users?id={broadcaster_id}"
                headers = {
                    'Authorization': f'Bearer {get_twitch_oauth_token()}',
                    'Client-Id': twitch_client_id
                }
                try:
                    response = requests.get(api_endpoint, headers=headers) 
                    response.raise_for_status()         
                    response_data = response.json()
                    display_name = response_data['data'][0]['display_name']
                    if len(display_name) > 0:
                        if subscription_type == "stream.online":
                            day_of_week, current_time = get_day_and_time()
                            append_to_csv(display_name, day_of_week, current_time)
                            db_user = database.get_user_by_twitch_channel(display_name)
                            if db_user:
                                user_id = db_user['id']
                                db_ref = db.reference(f'users/{user_id}')
                                db_ref.update({'displayHome': True, 'twitchProfile': display_name })
                        elif subscription_type == "stream.offline":
                            db_user = database.get_user_by_twitch_channel(display_name)
                            if db_user:
                                user_id = db_user['id']
                                db_ref = db.reference(f'users/{user_id}')
                                db_ref.update({'displayHome': False})
                except requests.exceptions.HTTPError as http_err:
                    return f"HTTP error occurred: {http_err}"
                except Exception as err:
                    return f"Other error occurred: {err}"
    return '', 204

@routes_bp.route('/match_template_spectating/<string:event_id>', methods=['POST'])
def match_template_spectating_route(event_id):
    user_id = request.json.get('userId')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User ID is required.'}), 400

    twitch_channel = request.json.get('twitchProfile')
    if not twitch_channel:
        return jsonify({'status': 'error', 'message': 'Twitch Username is required.'}), 400

    online_status = check_user_online(twitch_channel)
    status = online_status.get('status')
    
    if status == 'online':
        logging.info(f"Starting process recording for {twitch_channel}")
        recorder = TwitchRecorder(twitch_channel, event_id, user_id)
        recorder.process_warzone_video_stream_info()  # Starts the thread

        # Immediately respond that the process has started
        return jsonify({
            'status': 'success',
            'message': 'Video processing has started.',
            'details': status
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'You should be a part of this match event and be streaming on your Twitch account.',
            'details': status
        })
    
@routes_bp.route('/heatmap', methods=['GET'])
def generate_and_serve_heatmap():
    base_path = os.path.dirname(__file__)  
    csv_path = os.path.join(base_path, '..', '..', 'warzone-streamer.csv')
    heatmap_path = os.path.join(base_path, '..', '..', 'heatmap.png')
    
    if not os.path.exists(csv_path):
        return jsonify({'message': 'CSV file not found'}), 404
    
    df = pd.read_csv(csv_path)

    if df.empty:
        return jsonify({'message': 'CSV file is empty, no data to generate heatmap'}), 400

    generate_heatmap(df, heatmap_path)
    
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/png')
    else:
        return jsonify({'message': 'Failed to generate heatmap'}), 500

# Route to trigger stream processing
@routes_bp.route('/api/process-stream', methods=['POST'])
def process_stream():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    user_Id = data.get('userId')
    if not user_Id:
        return jsonify({'error': 'User Id is required'}), 400
    
    
    event_Id = data.get('eventId')
    if not event_Id:
        return jsonify({'error': 'Event Id is required'}), 400
    
    match_duration = database.get_match_duration_by_event_id(event_Id)

    if not match_duration:
        return jsonify({'error': 'Match Limit is required'}), 400
    
    try:
        access_token = get_twitch_access_token()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    twitch_api_url = f"https://api.twitch.tv/helix/streams?user_login={username}"
    headers = {
        'Client-ID': TWITCH_CLIENT_ID,
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(twitch_api_url, headers=headers)
    if response.status_code != 200 or not response.json()['data']:
        return jsonify({'error': f'{username} is not live on Twitch'}), 409

    try:
        # Enqueue the task to Celery
        task = process_twitch_stream.delay(username, user_Id, event_Id, match_duration)
        return jsonify({'task_id': str(task.id), 'status': 'Processing started'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
 
                          
def start_scheduler_thread():
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

start_scheduler_thread()
