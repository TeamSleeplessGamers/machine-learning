from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS
from dotenv import load_dotenv
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import pytesseract
import os
import cv2
import cv2 as cv
import psycopg2
import subprocess
import logging
import shutil
import enum
import threading

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app, origins='*')  # Enable CORS for all origins
app.secret_key = os.urandom(24)

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('STREAM_LABS_REDIRECT')
token_url = 'https://streamlabs.com/api/v2.0/token'
oauth_url = 'https://streamlabs.com/api/v2.0/authorize'
database_url = os.getenv('DATABASE_URL')
vision_kill_skull = Vision('./game_templates/warzone/interest_3.jpg')

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode
chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Global database connection
conn = None

# skull HSV filter
hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)

def initialize_database():
    global conn
    try:
        conn = psycopg2.connect(database_url)
        print("Database connection established successfully.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        conn = None

class TwitchResponseStatus(enum.Enum):
    ONLINE = 0
    OFFLINE = 1
    NOT_FOUND = 2
    UNAUTHORIZED = 3
    ERROR = 4

class TwitchRecorder:
    def __init__(self, username):
        # global configuration
        self.ffmpeg_path = "ffmpeg"
        self.disable_ffmpeg = False
        self.refresh = 15
        self.root_path = "./"

        # user configuration
        self.username = username
        self.quality = "best"

        # twitch configuration
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = f"https://id.twitch.tv/oauth2/token?client_id={self.client_id}&client_secret={self.client_secret}&grant_type=client_credentials"
        self.url = "https://api.twitch.tv/helix/streams"
        self.access_token = self.fetch_access_token()
    def fetch_access_token(self):
        token_response = requests.post(self.token_url, timeout=15)
        token_response.raise_for_status()
        token = token_response.json()
        return token["access_token"]

    def run(self):
        # path to recorded stream
        recorded_path = os.path.join(self.root_path, "recorded", self.username)
        # path to finished video, errors removed
        processed_path = os.path.join(self.root_path, "processed", self.username)

        # create directory for recordedPath and processedPath if not exist
        if os.path.isdir(recorded_path) is False:
            os.makedirs(recorded_path)
        if os.path.isdir(processed_path) is False:
            os.makedirs(processed_path)

        # make sure the interval to check user availability is not less than 15 seconds
        if self.refresh < 15:
            logging.warning("check interval should not be lower than 15 seconds")
            self.refresh = 15
            logging.info("system set check interval to 15 seconds")

        logging.info("checking for %s every %s seconds, recording with %s quality",
                     self.username, self.refresh, self.quality)
        self.loop_check(recorded_path, processed_path)
    def record_stream(self, recorded_filename):
        subprocess.call(
            ["streamlink", "--twitch-disable-ads", "twitch.tv/" + self.username, self.quality,
             "-o", recorded_filename])

    def process_recorded_file(self, recorded_filename, processed_filename):
        if os.path.exists(processed_filename):
            os.remove(processed_filename)  # Delete existing processed file if it exists

        if self.disable_ffmpeg:
            logging.info("moving: %s", recorded_filename)
            shutil.move(recorded_filename, processed_filename)
        else:
            logging.info("fixing %s", recorded_filename)
            self.ffmpeg_copy_and_fix_errors(recorded_filename, processed_filename)
        # After processing, use OpenCV VideoCapture to work with the processed video file
        #self.save_processed_frames(processed_filename)
    def ffmpeg_copy_and_fix_errors(self, recorded_filename, processed_filename):
        try:
            subprocess.call(
                [self.ffmpeg_path, "-err_detect", "ignore_err", "-i", recorded_filename, "-c", "copy",
                 processed_filename])
            os.remove(recorded_filename)
        except Exception as e:
            logging.error(e)
    def save_processed_frames(self, processed_filename):
        cap = cv2.VideoCapture(processed_filename, cv2.CAP_FFMPEG)

        output_folder = "./processed_frames"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_count = 0
        template = cv2.imread('./game_templates/warzone/interest_1.png', 0)  # Load your template image
        template_width, template_height = template.shape[::-1]  # Template width and height
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ensure frame is valid
            if frame is None:
                continue
            
            # do object detection
            #rectangles = vision_limestone.find(processed_image, 0.46)

            # draw the detection results onto the original image
            #output_image = vision_limestone.draw_rectangles(screenshot, rectangles)
            # Process each frame as needed (convert to grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            output_filename = os.path.join("./test_images_gray_frame", f"frame_{frame_count}.jpg")
            cv2.imwrite(output_filename, thresh)
            
            # Perform template matching
            if template is None:
                continue
            
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            # Draw a rectangle around the detected area
            x, y, width, height = 1800, 40, 50, 50
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green rectangle, thickness 2
            # Extract the ROI using the rectangle's coordinates
            roi = frame[y:y+height, x:x+width]
  
            # Example: Use OCR (e.g., pytesseract) to extract text from the ROI
            text = pytesseract.image_to_string(roi, config='--psm 13')
            
            if len(text) > 0:
                print(f"Text detected in ROI: {frame_count} - {text}")
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


    def check_user(self):
        info = None
        status = TwitchResponseStatus.ERROR
        try:
            headers = {"Client-ID": self.client_id, "Authorization": "Bearer " + self.access_token}
            r = requests.get(self.url + "?user_login=" + self.username, headers=headers, timeout=15)
            r.raise_for_status()
            info = r.json()
            if info is None or not info["data"]:
                status = TwitchResponseStatus.OFFLINE
            else:
                status = TwitchResponseStatus.ONLINE
        except requests.exceptions.RequestException as e:
            if e.response:
                if e.response.status_code == 401:
                    status = TwitchResponseStatus.UNAUTHORIZED
                if e.response.status_code == 404:
                    status = TwitchResponseStatus.NOT_FOUND
        return status, info

    def loop_check(self, recorded_path, processed_path):
        while True:
            status, info = self.check_user()
            if status == TwitchResponseStatus.NOT_FOUND:
                logging.error("username not found, invalid username or typo")
                time.sleep(self.refresh)
            elif status == TwitchResponseStatus.ERROR:
                logging.error("%s unexpected error. will try again in 5 minutes",
                              datetime.now().strftime("%Hh%Mm%Ss"))
                time.sleep(300)
            elif status == TwitchResponseStatus.OFFLINE:
                logging.info("%s currently offline, checking again in %s seconds", self.username, self.refresh)
                time.sleep(self.refresh)
            elif status == TwitchResponseStatus.UNAUTHORIZED:
                logging.info("unauthorized, will attempt to log back in immediately")
                self.access_token = self.fetch_access_token()
            elif status == TwitchResponseStatus.ONLINE:
                logging.info("%s online, stream recording in session", self.username)
                filename = self.username + ".mp4"
                recorded_filename = os.path.join(recorded_path, filename)
                processed_filename = os.path.join(processed_path, filename)

                # Start recording in a new thread
                #record_thread = threading.Thread(target=self.record_stream, args=(recorded_filename,))
                #record_thread.start()

                # Start processing in a new thread
                #process_thread = threading.Thread(target=self.process_recorded_file, args=(recorded_filename, processed_filename))
                #process_thread.start()

                logging.info("processing and recording started, going back to checking...")
                time.sleep(self.refresh)


# Routes
@app.route('/')
def index():
    return '''
        <h1>Welcome to Stream Integration</h1>
        <a href="/authorize_streamlabs">Authorize with Streamlabs</a>
    '''

@app.route('/authorize_streamlabs')
def authorize_streamlabs():
    userid = request.args.get('userid') or ""
    auth_url = f'{oauth_url}?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=mediashare.control&state={userid}'
    return jsonify({'auth_url': auth_url})

@app.route('/callback')
def callback():
    initialize_database()
    code = request.args.get('code')
    userid = request.args.get('state')
    response = requests.post(token_url, data={
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'redirect_uri': redirect_uri
    })

    token_data = response.json()
    session['streamlabs_token'] = token_data['access_token']
    
    # Insert the token into the database with userid and created_at timestamp
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO user_streamlabs_tokens (user_id, streamlabs_token, created_at) VALUES (%s, %s, %s)",
                           (userid, token_data['access_token'], current_time))
            conn.commit()
    except Exception as e:
        print(f"Error inserting token into database: {e}")
        return jsonify({'error': 'Failed to insert token into database'}), 500

    return 'Streamlabs authorization successful!'

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # Open the webpage
        driver.get(url)

        # Wait for the page to load and JavaScript to execute
        time.sleep(5)  # Adjust this delay as needed

        # Find all iframe elements
        iframe_elements = driver.find_elements(By.TAG_NAME, 'iframe')

        # Extract src attribute values from iframe elements
        twitch_urls = []
        for iframe in iframe_elements:
            src = iframe.get_attribute('src')
            if src.startswith('https://player.twitch.tv'):
                twitch_urls.append(src)

        return jsonify({'twitch_urls': twitch_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def match_template_in_video(video_path, template_path, output_folder, threshold=0.95, save_as_images=True):
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

        # Crop the frame to the top-right corner with height 100 pixels
        height = 200
        width = int((height / frame.shape[0]) * frame.shape[1])
        x_start = frame.shape[1] - width
        y_start = 0
        cropped_frame = frame[y_start:y_start + height, x_start:x_start + width]

        # Check if the template size is appropriate for the cropped frame
        if (template_height > cropped_frame.shape[0]):
            print("Warning: Template is larger than the cropped frame. Skipping this frame.")
            continue

        # Convert the template to match the frame's type and depth
        if len(cropped_frame.shape) == 3 and len(template.shape) == 2:  # Color frame and grayscale template
            template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

        # Ensure both cropped_frame and template have the same depth and type
        if cropped_frame.dtype != template.dtype:
            template = template.astype(cropped_frame.dtype)

        # Apply template matching
        result = cv2.matchTemplate(cropped_frame, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("Max value:", max_val)

        # Initialize variables
        top_left = None
        bottom_right = None

        # Check if the best match is above the threshold
        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
            print(frame_count, "frame here", top_left, "what is template width", bottom_right)
            # Draw a rectangle around the matched region if a match was found
            if top_left is not None and bottom_right is not None:
                cv2.rectangle(cropped_frame, top_left, bottom_right, (0, 255, 0), 2)

            # Save the frame if the threshold is met
            if save_as_images:
                output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_filename, cropped_frame)
            else:
                out.write(cropped_frame)

    print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    if not save_as_images:
        out.release()
    cv2.destroyAllWindows()

@app.route('/match_template', methods=['POST'])
def match_template_route():
    output_folder = "./test_video"

    # Call the match_template_in_video function
    match_template_in_video("./processed/xressolve/xressolve.mp4", "./game_templates/warzone/interest_1.jpg", output_folder)

    return jsonify({'status': 'success', 'message': 'Processing completed.'})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        recorder = TwitchRecorder(username)
        threading.Thread(target=recorder.run).start()
        return jsonify({'message': 'Recording started for user: ' + username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the application
if __name__ == "__main__":
    logging.basicConfig(filename="twitch-recorder.log", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        app.run(debug=True, host='0.0.0.0', port=8000)
    finally:
        driver.quit()
        if conn:
            conn.close()
