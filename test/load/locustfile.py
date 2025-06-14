from locust import HttpUser, task, between
import json
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WebsiteUser(HttpUser):
    wait_time = between(2, 10)  # seconds between tasks
    
    twitch_users = [
        "nadeshot", "nadeshot", "nadeshot", "nadeshot", "nadeshot",
        "nadeshot", "nadeshot", "nadeshot", "nadeshot", "nadeshot"
    ]

    @task
    def index(self):
        self.client.get("/")

    @task
    def process_stream(self):
        payload = {
            "username": random.choice(self.twitch_users),
            "userId": random.randint(1, 1000),
            "eventId": "143"
        }

        headers = {"Content-Type": "application/json"}

        response = self.client.post(
            "/api/process-stream",
            data=json.dumps(payload),
            headers=headers
        )
        
        print("Status Code:", response.status_code)
        print("Response Body:", response.text)
