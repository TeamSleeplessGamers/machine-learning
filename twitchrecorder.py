import time
from datetime import datetime
import requests
import pytesseract
import os
import cv2
import subprocess
import logging
import enum
import threading
from warzone import match_template_spectating_in_video

class TwitchResponseStatus(enum.Enum):
    ONLINE = 0
    OFFLINE = 1
    NOT_FOUND = 2
    UNAUTHORIZED = 3
    ERROR = 4

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
 
class TwitchRecorder:
    def __init__(self, username, event_id=None, user_id=None):
        # global configuration
        self.username = username
        self.event_id = event_id
        self.user_id = user_id
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
        if self.refresh < 15:
            logging.warning("check interval should not be lower than 15 seconds")
            self.refresh = 15
            logging.info("system set check interval to 15 seconds")
        logging.info("checking for %s every %s seconds, recording with %s quality",
                     self.username, self.refresh, self.quality)
        self.loop_check()
    def record_stream(self, recorded_filename):
        subprocess.call(
            ["streamlink", "--twitch-disable-ads", "twitch.tv/" + self.username, self.quality,
             "-o", recorded_filename])

    def get_live_stream_url(self, twitch_profile):
        try:
            result = subprocess.run(
                ["streamlink", "--twitch-disable-ads", f"https://twitch.tv/{twitch_profile}", "best", "--stream-url"],
                check=True, text=True, capture_output=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return None
    def process_recorded_file(self):
        stream_url = self.get_live_stream_url(self.username)
        self.processed_match_template_spectating(stream_url)
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
    def processed_match_template_spectating(self, processed_filename):
            match_template_spectating_in_video(processed_filename, event_id=self.event_id, user_id=self.user_id)
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

    def loop_check(self):
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
                
                # Start processing in a new thread
                process_thread = threading.Thread(target=self.process_recorded_file)
                process_thread.start()

                logging.info("processing and recording started, going back to checking...")
                time.sleep(self.refresh)
