import schedule
import time
from datetime import datetime, timedelta
import os
import requests
from .twitch_oauth import get_twitch_oauth_token
from ..config.database import Database
from dotenv import dotenv_values


database = Database()
conn = database.get_connection()
 
def list_of_sg_subscribed_twitch_streamers():
    twitch_client_id = os.environ.get('CLIENT_ID')
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
    twitch_client_id = os.environ.get('CLIENT_ID')
    twitch_webhook_url = os.environ.get('TWITCH_WEBHOOK_URL')
    twitch_client_secret = os.environ.get('CLIENT_SECRET')
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
    twitch_client_id = os.environ.get('CLIENT_ID')

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
        

def create_gpu_task_for_event(event_id): 
    print("handle creating gpu")

    url = "https://rest.runpod.io/v1/pods"
    env = dotenv_values(".env")
    
    payload = {
        "allowedCudaVersions": ["12.8"],
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": 50,
        "containerRegistryAuthId": "clzdaifot0001l90809257ynb",
        "countryCodes": ["US"],
        "cpuFlavorPriority": "availability",
        "dataCenterIds": ["US-DE-1"],
        "dataCenterPriority": "availability",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "gpuTypeIds": ["NVIDIA L4"], 
        "vcpuCount": 12,             
        "minRAMPerGPU": 55,            
        "volumeInGb": 20,     
        "env": env,
        "globalNetworking": True,
        "gpuCount": 2,
        "gpuTypePriority": "availability",
        "interruptible": False,
        "locked": False,
        "minDiskBandwidthMBps": 123,
        "minDownloadMbps": 123,
        "minUploadMbps": 123,
        "minVCPUPerGPU": 2,
        "name": "event-pod-{event_id}",
        "networkVolumeId": "",
        "ports": ["8888/http", "22/tcp"],
        "supportPublicIp": True,
        "templateId": "",
        "volumeInGb": 20,
        "volumeMountPath": "/workspace"
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    print("RunPod API response:", response.status_code, response.text)

    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        pod_id = data.get("id")
        print(f"Pod created with ID: {pod_id}")
        return pod_id
    else:
        print("Failed to create pod")
        return None
    
def hourly_job():
    events = database.get_match_events_for_today()
    
    if len(events) > 0:
        for event in events:
            start_time = event['start_date']
            time_now = datetime.now(start_time.tzinfo)

            time_diff = start_time - time_now

            if timedelta(minutes=0) < time_diff <= timedelta(hours=1):
                if database.is_gpu_task_running_for_entity("event", event['id']):
                    return
                else:
                    results = create_gpu_task_for_event(event['id'])
                    print(f"Then what is results {results}")
                    # database.update_gpu_task_for_event_start_soon("event", event['id'], "pod_id", "running")
            else:
                print(f"[{datetime.now()}] Skipping Event ID {event['id']}, starts in {time_diff}. Not in 1-hour window.")
    else:
        print(f"[{datetime.now()}] No events found today.")

def start_hourly_scheduler():
    # Update to be every 15 minutes
    schedule.every(15).seconds.do(hourly_job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    
