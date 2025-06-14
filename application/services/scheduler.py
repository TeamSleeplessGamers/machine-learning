import schedule
import time
from datetime import datetime, timedelta
import os
import requests
from .twitch_oauth import get_twitch_oauth_token
from ..config.database import Database
from dotenv import dotenv_values
import redis

database = Database()
conn = database.get_connection()
r = redis.Redis.from_url(os.environ.get("REDIS_URL"))

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
    url = "https://rest.runpod.io/v1/pods"
    env = dotenv_values(".env")
    
    payload = {
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypePriority": "availability",
        "cloudType": "SECURE",
        "cpuFlavorPriority": "availability",
        "dataCenterPriority": "availability",
        "countryCodes": ["US"],
        "vcpuCount": 6,
        "minRAMPerGPU": 24,
        "containerDiskInGb": 40,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
         "dockerEntrypoint": ["bash", "run_gpu_setup.sh"],
        "imageName": "mjubil1/sleepless-gpu-worker:latest",
        "env": env,
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "name": f"event-pod-{event_id}",
        "supportPublicIp": True,
        "interruptible": False,
        "globalNetworking": True,
        "locked": False,
        "ports": ["8888/http", "22/tcp"]
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        pod_id = data.get("id")
        return pod_id
    else:
        return None

def schedule_shutdown_for_event(event_id, pod_id, time_limit_minutes):
    time_limit = timedelta(minutes=time_limit_minutes)
    start_time = datetime.now()

    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        remaining = time_limit - elapsed

        if remaining.total_seconds() <= 0:
            print(f"[{datetime.now()}] Time is up for Event {event_id}, shutting down.")
            shutdown_gpu_task(event_id, pod_id)
            break

def shutdown_gpu_task(event_id, pod_id):
    headers = {"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY')}"}

    try:
        terminate_response = requests.delete(
            f"https://rest.runpod.io/v1/pods/{pod_id}",
            headers=headers
        )
        print(f"[{datetime.now()}] Terminate response: {terminate_response.status_code} - {terminate_response}")

        if terminate_response.status_code == 200:
            print(f"✅ Successfully terminated pod {pod_id} for event {event_id}.")
        else:
            print(f"❌ Failed to terminate pod {pod_id}: {terminate_response.text}")

    except Exception as e:
        print(f"⚠️ Error shutting down pod {pod_id}: {e}")

    try:
        database.update_gpu_task_for_event_start_soon(
            "event", event_id, pod_id, "stopped", stopped_at=datetime.utcnow()
        )
        print(f"📦 Updated database: pod {pod_id} marked as stopped.")
    except Exception as db_error:
        print(f"❗ Error updating DB for pod {pod_id}: {db_error}")

def fifteen_minute_job():
    events = database.get_match_events_for_today()
    
    if len(events) > 0:
        for event in events:
            start_time = event['start_date']
            time_now = datetime.now(start_time.tzinfo)

            time_diff = start_time - time_now
            is_gpu_task_is_running_for_entity = database.is_gpu_task_running_for_entity("event", event['id'])
            if timedelta(minutes=0) < time_diff <= timedelta(hours=1):
                is_running = database.is_gpu_task_running_for_entity("event", event['id'])
                
                lock_key = f"gpu_lock:{event['id']}"
                acquired = r.set(lock_key, "1", nx=True, ex=15)  
                if acquired and is_running is False:
                    result = create_gpu_task_for_event(event['id'])
                    if result:
                        database.update_gpu_task_for_event_start_soon("event", event['id'], result, "running")
                    return

            if is_gpu_task_is_running_for_entity:
                pod_id = database.get_gpu_pod_id_for_entity("event", event['id'])
                if pod_id and event['time_limit'] is not None:
                    time_limit_minutes = event['time_limit']
                    schedule_shutdown_for_event(event['id'], pod_id, time_limit_minutes)
                    return
    else:
        return

def start_fifteen_minute_scheduler():
    schedule.every(15).seconds.do(fifteen_minute_job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    
