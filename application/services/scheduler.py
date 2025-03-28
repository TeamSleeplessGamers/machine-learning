import schedule
import time
from datetime import datetime, timedelta
import os
import requests
from .twitch_oauth import get_twitch_oauth_token
from ..config.database import Database


database = Database()
conn = database.get_connection()
 
def list_of_sg_subscribed_twitch_streamers():
    twitch_client_id = os.environ['CLIENT_ID']
    headers = {
        'Authorization': f'Bearer {get_twitch_oauth_token()}',
        'Client-Id': twitch_client_id
    }
    broadcaster_user_ids = []
    url = 'https://api.twitch.tv/helix/eventsub/subscriptions'

    while url:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            broadcaster_user_ids.extend(
                item['condition']['broadcaster_user_id']
                for item in response_data['data']
                if item['cost'] == 1
            )
            url = response_data.get('pagination', {}).get('cursor')
            if url:
                url = f'https://api.twitch.tv/helix/eventsub/subscriptions?after={url}'

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            break
        except Exception as err:
            print(f"Other error occurred: {err}")
            break
    return broadcaster_user_ids
                
def process_user_id_to_subscribe(user_ids):
    twitch_client_id = os.environ['CLIENT_ID']
    twitch_webhook_url = os.environ['TWITCH_WEBHOOK_URL']
    twitch_client_secret = os.environ['CLIENT_SECRET']
    event_types = ["stream.online", "stream.offline"]

    for user_id in user_ids:
        for event_type in event_types:
            headers = {
                'Authorization': f'Bearer {get_twitch_oauth_token()}',
                'Client-Id': twitch_client_id,
            }

            payload = {
                "type": event_type,
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

            try:
                response = requests.post('https://api.twitch.tv/helix/eventsub/subscriptions', headers=headers, json=payload)
                response.raise_for_status()
                print(f"Successfully subscribed to {event_type} for user {user_id}")
                time.sleep(10)
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
            except requests.exceptions.RequestException as req_err:
                print(f"Request error occurred: {req_err}")
            except Exception as err:
                print(f"An unexpected error occurred: {err}")
             
def subscribe_to_twitch_streamers():
    twitch_client_id = os.environ['CLIENT_ID']

    users = database.get_list_of_sg_users()
    if len(users) > 0:
        print(f"Total users found: {len(users)}")
        
        subscribed_streamers = list_of_sg_subscribed_twitch_streamers()
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
                existing_subscriber_user_ids = set(subscribed_streamers)
                user_ids = [user['id'] for user in response_data['data'] if user['id'] not in existing_subscriber_user_ids]
                if len(user_ids) > 0:
                    process_user_id_to_subscribe(user_ids)

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
            except Exception as err:
                print(f"Other error occurred: {err}")
        print("Finished Processing User Info.")

def start_scheduler():
    now = datetime.now()
    target_time = now.replace(hour=9, minute=0, second=0, microsecond=0)

    if now > target_time:
        target_time += timedelta(days=1)

    schedule.every().day.at(target_time.strftime("%H:%M:%S")).do(subscribe_to_twitch_streamers)

    while True:
        schedule.run_pending()
        time.sleep(1)
    
