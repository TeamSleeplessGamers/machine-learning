from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS
from dotenv import load_dotenv
from vision import Vision
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
        #if os.path.exists(processed_filename):
        #    os.remove(processed_filename)  # Delete existing processed file if it exists

        if self.disable_ffmpeg:
            logging.info("moving: %s", recorded_filename)
            shutil.move(recorded_filename, processed_filename)
        else:
            logging.info("fixing %s", recorded_filename)
            self.ffmpeg_copy_and_fix_errors(recorded_filename, processed_filename)
        # After processing, use OpenCV VideoCapture to work with the processed video file
        self.save_processed_frames(processed_filename)
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
            
            # Extract top-left and bottom-right coordinates of the detected area
            top_left = max_loc
            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
            
            # Draw a rectangle around the detected area
            x, y, width, height = 1800, 20, 50, 100
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green rectangle, thickness 2

            # Save the processed frame with the rectangle to the output folder
            output_filename = os.path.join("./processed_frames", f"frame_{frame_count}.jpg")
            cv2.imwrite(output_filename, frame)
            
            # Extract the ROI using the rectangle's coordinates
            roi = frame[y:y+height, x:x+width]

            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast (optional)
            enhanced_roi = cv2.equalizeHist(gray_roi)
            
            points = vision_limestone.find(screenshot, 0.5, 'rectangles')
            # Apply adaptive thresholding to create a binary image
            binary_roi = cv2.adaptiveThreshold(enhanced_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Save the ROI image to a folder for inspection (binary image)
            roi_output_filename = os.path.join('./roi_frames', f"roi_{frame_count}.jpg")
            cv2.imwrite(roi_output_filename, binary_roi)
            
            # Example: Use OCR (e.g., pytesseract) to extract text from the ROI
            text = pytesseract.image_to_string(binary_roi, config='outputbase digits')
            print(f"Text detected in ROI: {text}")
            
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
                #time.sleep(self.refresh)
            elif status == TwitchResponseStatus.ERROR:
                logging.error("%s unexpected error. will try again in 5 minutes",
                              datetime.now().strftime("%Hh%Mm%Ss"))
                #time.sleep(300)
            elif False: #status == TwitchResponseStatus.OFFLINE:
                logging.info("%s currently offline, checking again in %s seconds", self.username, self.refresh)
                #time.sleep(self.refresh)
            elif status == TwitchResponseStatus.UNAUTHORIZED:
                logging.info("unauthorized, will attempt to log back in immediately")
                self.access_token = self.fetch_access_token()
            elif True: #status == TwitchResponseStatus.ONLINE:
                logging.info("%s online, stream recording in session", self.username)
                filename = self.username + ".mp4"
                recorded_filename = os.path.join(recorded_path, filename)
                processed_filename = os.path.join(processed_path, filename)

                # Start recording in a new thread
                #record_thread = threading.Thread(target=self.record_stream, args=(recorded_filename,))
                #record_thread.start()

                # Start processing in a new thread
                process_thread = threading.Thread(target=self.process_recorded_file, args=(recorded_filename, processed_filename))
                process_thread.start()

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
