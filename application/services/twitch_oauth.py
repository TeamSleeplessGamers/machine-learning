import os
import requests

def get_twitch_oauth_token():
    twitch_client_id = os.environ['CLIENT_ID']
    twitch_client_secret = os.environ['CLIENT_SECRET']
    twitch_oauth_url = os.environ['TWITCH_OAUTH_URL']

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