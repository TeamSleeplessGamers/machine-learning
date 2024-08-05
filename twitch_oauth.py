import os
import requests

twitch_client_id = os.getenv('CLIENT_ID')
twitch_client_secret = os.getenv('CLIENT_SECRET')
twitch_webhook_url = os.getenv('TWITCH_WEBHOOK_URL')
twitch_oauth_url = os.getenv('TWITCH_OAUTH_URL')

def get_twitch_oauth_token():
    params = {
        'client_id': twitch_client_id,
        'client_secret': twitch_client_secret,
        'grant_type': 'client_credentials'
    } 
    response = requests.post(twitch_oauth_url, params=params)
    response_data = response.json()

    if response.status_code == 200 and 'access_token' in response_data:
        return response_data['access_token']
    else:
        return None