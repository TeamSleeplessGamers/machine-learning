import time
from datetime import datetime
import requests
import pytesseract
import os
import cv2
import subprocess
import logging
import threading
import enum
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
    def __init__(self, username):
        # global configuration
        self.username = username
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
    def process_warzone_video_stream_info(self):
        stream_url = self.get_live_stream_url(self.username)
        # Run the video processing function in a separate thread
        processing_thread = threading.Thread(
            target=match_template_spectating_in_video,
            args=(stream_url, self.event_id, self.user_id)
        )
        processing_thread.start()
                

    