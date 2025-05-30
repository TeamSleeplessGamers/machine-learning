from datetime import datetime
from .twitch_recorder import TwitchRecorder
from .warzone import pull_warzone_stream_and_process
from ..celery_config import celery_app
from ..config.firebase import initialize_firebase
from dotenv import load_dotenv
from .warzone import init_data
import multiprocessing
import logging

load_dotenv()

@celery_app.task(bind=True, name="process_warzone_video_stream_info")
def process_warzone_video_stream_info_task(self, username, event_id, user_id, team_id):
    multiprocessing.set_start_method('spawn', force=True)
    initialize_firebase()
    recorder = TwitchRecorder(username, event_id, user_id, team_id)
    stream_url = recorder.get_live_stream_url()
    # Use the current time as the start time
    start_datetime = datetime.now()

    if not stream_url:
        raise ValueError(f"Stream URL for {username} could not be retrieved.")

    try:
        init_data(event_id, user_id, team_id)
    except Exception as e:
        logging.error(f"Failed to initialize Firebase match data: {e}")
        raise self.retry(exc=e, countdown=10, max_retries=3)

    try:
        pull_warzone_stream_and_process(start_datetime, stream_url, event_id, user_id, team_id)
    except Exception as e:
        raise self.retry(exc=e, countdown=10, max_retries=3)