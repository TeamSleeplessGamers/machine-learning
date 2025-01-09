from flask import Blueprint, request, jsonify, send_file, Response
import pandas as pd
import time
from ..services.twitch_recorder import TwitchRecorder
from ..services.twitch_oauth import get_twitch_oauth_token
from ..services.machine_learning import detect_text_with_api_key
import pytesseract
import os
from firebase_admin import db
import cv2
import hashlib
import pytz
import hmac
import logging
from ..utils.heatmap_generator import generate_heatmap
import requests
import csv
import threading
from ..services.scheduler import start_scheduler
from datetime import datetime
from ..config.database import Database

database = Database()
conn = database.get_connection()

threshold = 10 
routes_bp = Blueprint('routes', __name__)

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
        logging.info(f"Starting Process Recoding for {twitch_channel}")
        recorder = TwitchRecorder(twitch_channel, event_id, user_id)
        recorder.process_warzone_video_stream_info()
    else:
        return jsonify({
            'message': 'Twitch User Must Not Be Online?',
            'details': status
        })              
    return jsonify({
        'message': 'Twitch User Online Status',
        'details': status
    })
    
@routes_bp.route('/read-image', methods=['POST'])
def read_image():
    """
    Endpoint to detect text from a hardcoded image file path.
    """
    # Hardcoded image path
    image_path = "/Users/trell/Projects/machine-learning/game_templates/warzone/shortcut_processed_frame_500.jpg"  # Update this path accordingly

    try:
        # Call the Vision API
        detected_texts = detect_text_with_api_key(image_path)

        return jsonify({"detected_texts": detected_texts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

def start_scheduler_thread():
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

start_scheduler_thread()
