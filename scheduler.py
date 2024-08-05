import schedule
import time
from datetime import datetime, timedelta
import os
import requests
from twitch_oauth import get_twitch_oauth_token
from database import Database
import signal
import sys

twitch_client_id = os.getenv('CLIENT_ID')
twitch_client_secret = os.getenv('CLIENT_SECRET')
twitch_webhook_url = os.getenv('TWITCH_WEBHOOK_URL')
twitch_oauth_url = os.getenv('TWITCH_OAUTH_URL')

database = Database()
conn = database.get_connection()
                
def subscribe_to_twitch_streamers():
    users = database.get_list_of_sg_users()
    if len(users) > 0:
        print(f"Total users found: {len(users)}")
        
        batch_size = 50
        for i in range(0, len(users), batch_size):
            batch = users[i:i + batch_size]
            cleaned_batch = [user.replace(' ', '') for user in batch]
            query_string = '&'.join(f'login={user}' for user in cleaned_batch)
            api_endpoint = f"https://api.twitch.tv/helix/users?{query_string}"
            headers = {
                 'Authorization': f'Bearer {get_twitch_oauth_token()}',
                 'Client-Id': twitch_client_id
             }
            try:
                response = requests.get(api_endpoint, headers=headers) 
                response.raise_for_status()         
                response_data = response.json()
                print(response_data)
                
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
            except Exception as err:
                print(f"Other error occurred: {err}")

now = datetime.now()
target_time = now + timedelta(seconds=1)
time_difference = (target_time - now).seconds

schedule.every().day.at(target_time.strftime("%H:%M:%S")).do(subscribe_to_twitch_streamers)

def signal_handler(sig, frame):
    print("Shutting down...")
    database.close_connection()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

while True:
    schedule.run_pending()
    time.sleep(1)
    
