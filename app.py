from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
from firebase import initialize_firebase
import time
from twitch_recorder import TwitchRecorder
from collections import deque
import pytesseract
import os
import cv2
import psycopg2
import logging
from heatmap_generator import generate_heatmap
import requests

load_dotenv()

app = Flask(__name__)
CORS(app, origins='*')

initialize_firebase()
database_url = os.getenv('DATABASE_URL')
twitch_client_id = os.getenv('CLIENT_ID')
twitch_client_secret = os.getenv('CLIENT_SECRET')
twitch_webhook_url = os.getenv('TWITCH_WEBHOOK_URL')
twitch_oauth_url = os.getenv('TWITCH_OAUTH_URL')

conn = None

threshold = 10 

def initialize_database():
    global conn
    try:
        conn = psycopg2.connect(database_url)
        print("Database connection established successfully.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        conn = None

# Routes
@app.route('/')
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if 'mp4v' fails
    
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

        # Ensure both images have the same data type
        if gray_frame.dtype != gray_template.dtype:
            gray_template = gray_template.astype(gray_frame.dtype)

        # Ensure images are in the required depth (CV_8U or CV_32F)
        if gray_frame.dtype != 'uint8' and gray_frame.dtype != 'float32':
            gray_frame = gray_frame.astype('uint8')
        if gray_template.dtype != 'uint8' and gray_template.dtype != 'float32':
            gray_template = gray_template.astype('uint8')


        detected_text = pytesseract.image_to_string(gray_frame)
  
        # Search for the word "SPECTATING" in the detected text (case-insensitive)
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
            
            # Draw a rectangle around the matched region if a match was found
            if top_left is not None and bottom_right is not None:
                cv2.rectangle(gray_cropped_frame, top_left, bottom_right, (0, 255, 0), 2)

                test_roi = cropped_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                # Extract the region of interest (ROI) within the rectangle
                roi = gray_cropped_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                # Crop 50 pixels off the top of the ROI
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
  
@app.route('/match_template', methods=['POST'])
def match_template_route():
    output_folder = "./test_video"

    match_template_in_video("./processed/test_1.mp4", "./game_templates/warzone/spectating_1.jpg", output_folder)

    return jsonify({'status': 'success', 'message': 'Processing completed.'})

def get_twitch_oauth_token():
    params = {
        'client_id': twitch_client_id,
        'client_secret': twitch_client_secret,
        'grant_type': 'client_credentials'
    } 
    response = requests.post(twitch_oauth_url, params=params)
    response_data = response.json()

    if response.status_code == 200 and 'access_token' in response_data:
        return response_data['access_token']
    else:
        return None

def check_user_online(user_login):
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
    
@app.route('/match_template_spectating/<string:event_id>', methods=['POST'])
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
            
    return jsonify({
        'message': 'Twitch User Online Status',
        'details': online_status
    })
    
@app.route('/heatmap', methods=['GET'])
def generate_and_serve_heatmap():
    csv_path = 'warzone-streamer.csv'
    heatmap_path = 'heatmap.png'
    
    if not os.path.exists(csv_path):
        return jsonify({'message': 'CSV file not found'}), 404
    
    df = pd.read_csv(csv_path)
    
    generate_heatmap(df, heatmap_path)
    
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/png')
    else:
        return jsonify({'message': 'Failed to generate heatmap'}), 500

# Start the application
if __name__ == "__main__":
    logging.basicConfig(filename="twitch-recorder.log", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        app.run(debug=True, host='0.0.0.0', port=8000)
    finally:
        if conn:
            conn.close()
