import schedule
import time
from datetime import datetime, timedelta
import os
import requests
from .twitch_oauth import get_twitch_oauth_token
from ..config.database import Database
import redis
import math

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
        

def create_gpu_task_for_event(event_id, index):
    url = "https://rest.runpod.io/v1/pods"
    env = {
        "TWITCH_WEBHOOK_URL": "{{ RUNPOD_SECRET_TWITCH_WEBHOOK_URL }}",
        "TWITCH_OAUTH_URL": "{{ RUNPOD_SECRET_TWITCH_OAUTH_URL }}",
        "CLIENT_SECRET": "{{ RUNPOD_SECRET_CLIENT_SECRET }}",
        "CLIENT_ID": "{{ RUNPOD_SECRET_CLIENT_ID }}",
        "FIREBASE_DEV_JSON": "{{ RUNPOD_SECRET_FIREBASE_DEV_JSON }}",
        "PROJECT_NAME": "{{ RUNPOD_SECRET_PROJECT_NAME }}",
        "GPU_QUEUE_NAME": "{{ RUNPOD_SECRET_GPU_QUEUE_NAME }}",
        "PYTHON_VERSION": "{{ RUNPOD_SECRET_PYTHON_VERSION }}",
        "REDIS_PORT": "{{ RUNPOD_SECRET_REDIS_PORT }}",
        "REDIS_HOST": "{{ RUNPOD_SECRET_REDIS_HOST }}",
        "REDIS_URL": "{{ RUNPOD_SECRET_REDIS_URL }}",
        "FIREBASE_CRED_PATH": "{{ RUNPOD_SECRET_FIREBASE_CRED_PATH }}",
        "FIREBASE_KEY_BASE64": "{{ RUNPOD_SECRET_FIREBASE_KEY_BASE64 }}",
        "FIREBASE_DATABASE_URL": "{{ RUNPOD_SECRET_FIREBASE_DATABASE_URL }}",
        "DATABASE_URL": "{{ RUNPOD_SECRET_DATABASE_URL }}",
        "EVENT_ID": str(event_id)
    }
    
    payload = {
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypePriority": "availability",
        "cloudType": "SECURE",
        "cpuFlavorPriority": "availability",
        "dataCenterPriority": "availability",
        "countryCodes": ["US"],
        "vcpuCount": 16,     
        "supportPublicIp": True,
        "minRAMPerGPU": 24,
        "containerDiskInGb": 40,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "dockerEntrypoint": ["bash", "/opt/init/run_gpu_setup.sh"],
        "imageName": "mjubil1/sleepless-gpu-worker:v1.0.6",
        "env": env,
        "name": f"event-pod-{event_id}-{index}",
        "supportPublicIp": True,
        "interruptible": False,
        "locked": False,
        "ports": ["8888/http", "22/tcp"]
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code in (200, 201):
            data = response.json()
            pod_id = data.get("id")
            return pod_id, None
        else:
            error_msg = f"Error {response.status_code}: {response.text}"
            return None, error_msg
    except Exception as e:
        return None, f"Exception occurred: {str(e)}"

def schedule_shutdown_for_event(event_id, pod_id, time_limit_minutes, start_time):
    now = datetime.now(start_time.tzinfo) if start_time.tzinfo else datetime.now()
    time_diff = start_time - now
    
    # If event hasn't started yet, just return or do nothing (wait for next check)
    if time_diff > timedelta(minutes=0):
        print(f"Event {event_id} hasn't started yet. Time until start: {time_diff}")
        return
    
    # Event has started, check if time limit is reached
    elapsed = now - start_time
    if elapsed >= timedelta(minutes=time_limit_minutes):
        print(f"Time is up for Event {event_id}, shutting down pod {pod_id}.")
        shutdown_gpu_task(event_id, pod_id)
    else:
        remaining = timedelta(minutes=time_limit_minutes) - elapsed
        print(f"Event {event_id} running, {remaining} left.")

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
                    total_emails = 0
                    vcpu_per_server = 8
                    max_cpu_usage_per_server = 0.7
                    tasks_per_server = max_cpu_usage_per_server * vcpu_per_server

                    emails_value = event['emails']
                    if isinstance(emails_value, str):
                        emails_list = [email.strip() for email in emails_value.split(',') if email.strip()]
                    elif isinstance(emails_value, list):
                        emails_list = [email.strip() for email in emails_value if email.strip()]
                    else:
                        emails_list = []

                    total_emails = len(emails_list)
                      
                    number_of_servers = math.ceil(total_emails / tasks_per_server)
                    
                    for i in range(number_of_servers):
                        result, error = create_gpu_task_for_event(event['id'], i)
                        
                        if result:
                            database.update_gpu_task_for_event_start_soon("event", event['id'], result, "running")
                        else:
                            print(f"Failed to create GPU task for server #{i+1}: {error}")
                    return

            if is_gpu_task_is_running_for_entity:
                pod_ids = database.get_gpu_pod_ids_for_entity("event", event['id'])
                if pod_ids and event['time_limit'] is not None:
                    time_limit_minutes = event['time_limit']
                    for pod_id in pod_ids:
                        schedule_shutdown_for_event(event['id'], pod_id, time_limit_minutes, start_time)
                return

    else:
        return

def start_fifteen_minute_scheduler():
    schedule.every(15).minutes.do(fifteen_minute_job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    
