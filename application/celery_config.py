import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery = Celery(
    "application",
    broker=os.environ.get("REDIS_URL"),
    backend=os.environ.get("REDIS_URL")
)

# Optional: load additional settings from a separate config
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)
