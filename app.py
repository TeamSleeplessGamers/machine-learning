from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from hsvfilter import HsvFilter
from firebase import initialize_firebase
from twitchrecorder import TwitchRecorder
import time
from collections import deque
import pytesseract
import os
import cv2
import psycopg2
import logging
import requests

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app, origins='*')  # Enable CORS for all origins

# Initialize Firebase when the server starts
initialize_firebase()
database_url = os.getenv('DATABASE_URL')
twitch_client_id = os.getenv('CLIENT_ID')
twitch_client_secret = os.getenv('CLIENT_SECRET')
twitch_webhook_url = os.getenv('TWITCH_WEBHOOK_URL')
twitch_oauth_url = os.getenv('TWITCH_OAUTH_URL')

# Global database connection
conn = None

threshold = 10  # Number of frames where the word must be detected to confirm

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
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate a unique filename for the output video
    timestamp = int(time.time())
    output_video_path = os.path.join(output_folder, f'output_video_{timestamp}.mp4')

    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("Error: Cannot load template image.")
        return

    template_height, template_width = template.shape

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use the 'H264' codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if 'mp4v' fails
    
    if not save_as_images:
        # Initialize the video writer
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Cannot open video writer.")
            cap.release()
            return

    frame_count = 0

    # Process the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}")

        # Crop the frame to the top-right corner with height 200 pixels
        height = 200
        width = int((height / frame.shape[0]) * frame.shape[1])
        x_start = max(frame.shape[1] - width, 0)  # Ensure x_start is within bounds
        y_start = 0
        y_end = min(y_start + height, frame.shape[0])  # Ensure y_end is within bounds

        # Crop the frame
        cropped_frame = frame[y_start:y_end, x_start:frame.shape[1] ]

        # Resize the cropped frame by adding 500 to height and width
        new_height = min(cropped_frame.shape[0] + 500, frame.shape[0])
        new_width = min(cropped_frame.shape[1] + 500, frame.shape[1])

        resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

        # Convert the resized frame to grayscale
        gray_cropped_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Convert frame to grayscale if it is not already
        if frame.ndim == 3:  # Color image (BGR)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 4:  # Color image with alpha channel (BGRA)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:  # Already grayscale
            gray_frame = frame

        # Convert template to grayscale if it is not already
        if template.ndim == 3:  # Color image (BGR)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        elif template.ndim == 4:  # Color image with alpha channel (BGRA)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
        else:  # Already grayscale
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
            
        # Perform template matching
        test_result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(test_result)

        # Check if the template size is appropriate for the cropped frame
        if (template_height > gray_cropped_frame.shape[0]) or (template_width > gray_cropped_frame.shape[1]):
            print("Warning: Template is larger than the cropped frame. Skipping this frame.")
            continue

        # Apply template matching
        result = cv2.matchTemplate(gray_cropped_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Define a variable to reduce the width
        reduced_width = int(template_width * 0.7)  # Adjust the factor as needed
        reduced_height = int(template_height * 1.5)
        
        # Initialize variables
        top_left = None
        bottom_right = None

        # Check if the best match is above the threshold
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

                # Use Tesseract to extract text from the image
                numbers_string = pytesseract.image_to_string(gray_frame)
                print(frame_count, f"Detected text:")

            # Save the frame if the threshold is met
            if save_as_images:
                output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_filename, gray_frame)
            else:
                out.write(gray_cropped_frame)
    # Release resources
    cap.release()
    if not save_as_images:
        out.release()
    cv2.destroyAllWindows()
  
@app.route('/webhooks/callback', methods=['POST'])
def webhook_callback():
    data = request.json

    # Check if it's a verification request
    if 'challenge' in data:
        # Respond with the challenge token
        return data['challenge'], 200

    # Access event data from the dictionary
    event_data = data.get('event', {})
    
    # Print the event type and broadcaster user name if available
    event_type = event_data.get('type', 'Unknown')
    
    if event_type == 'live':
        broadcaster_user_name = event_data.get('broadcaster_user_name', 'Unknown')
        print(f"Event Type: {event_type}, Broadcaster User Name: {broadcaster_user_name}")

    return jsonify({'status': 'received'}), 200

@app.route('/match_template', methods=['POST'])
def match_template_route():
    output_folder = "./test_video"

    # Call the match_template_in_video function
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
    
def get_twitch_user_id(username):
    twitch_login_user_url = f'https://api.twitch.tv/helix/users?login={username}'
    headers = {
        'Client-ID': twitch_client_id,    
        'Authorization': f'Bearer {get_twitch_oauth_token()}'
    }
    response = requests.get(twitch_login_user_url, headers=headers)
    response_data = response.json()
    if response.status_code == 200 and 'data' in response_data and len(response_data['data']) > 0:
        return response_data['data'][0]['id']
    else:
        return None

def subscribe_to_events(user_id):
    # Define the subscription data for multiple event types
    subscriptions = [
        {
            "type": "stream.online",
            "version": "1",
            "condition": {
                "broadcaster_user_id": user_id
            },
            "transport": {
                "method": "webhook",
                "callback": twitch_webhook_url,
                "secret": twitch_client_secret
            }
        },
        {
            "type": "stream.offline",
            "version": "1",
            "condition": {
                "broadcaster_user_id": user_id
            },
            "transport": {
                "method": "webhook",
                "callback": twitch_webhook_url,
                "secret": twitch_client_secret
            }
        }
    ]
    
    headers = {
        'Client-ID': twitch_client_id,
        'Authorization': f'Bearer {get_twitch_oauth_token()}',
        'Content-Type': 'application/json'
    }
    
    responses = []
    twitch_subscription_url = 'https://api.twitch.tv/helix/eventsub/subscriptions'
    
    for subscription_data in subscriptions:
        response = requests.post(twitch_subscription_url, headers=headers, json=subscription_data)
        responses.append(response.json())
    
    return responses

def check_user_online(user_login):
    headers = {
        'Client-ID': twitch_client_id,    
        'Authorization': f'Bearer {get_twitch_oauth_token()}'
    }
    
    # Define the Twitch API URL to check if the user is online
    stream_info_url = f'https://api.twitch.tv/helix/streams?user_login={user_login}'
    
    # Make the request to Twitch API
    response = requests.get(stream_info_url, headers=headers)
    stream_data = response.json()
    
    # Check if the user is online
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

    # Check if the user is online
    online_status = check_user_online(twitch_channel)
    
    return jsonify({
        'message': 'Twitch User Online Status',
        'details': online_status
    })
    
# Start the application
if __name__ == "__main__":
    logging.basicConfig(filename="twitch-recorder.log", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        app.run(debug=True, host='0.0.0.0', port=8000)
    finally:
        if conn:
            conn.close()
