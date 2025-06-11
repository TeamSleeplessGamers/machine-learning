from locust import HttpUser, task, between
import json
import random

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # seconds between tasks

    @task
    def index(self):
        self.client.get("/")

    @task
    def process_stream(self):
        payload = {
            "username": f"testuser{random.randint(1,1000)}",
            "userId": random.randint(1, 1000),
            "eventId": "143"
        }

        headers = {"Content-Type": "application/json"}

        self.client.post(
            "/api/process-stream",
            data=json.dumps(payload),
            headers=headers
        )