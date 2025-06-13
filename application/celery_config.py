import os
import ssl
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

redis_url = os.environ.get("REDIS_URL")

celery = Celery(
    "application",
    broker=redis_url,
    backend=redis_url
)

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_REQUIRED
    }
)

from application.tasks import stream_task
