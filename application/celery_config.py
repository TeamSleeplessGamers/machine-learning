import os
import ssl
from celery import Celery
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

redis_url = os.environ.get("REDIS_URL")
is_production = os.environ.get("ENV") == "production"

if not redis_url:
    raise ValueError("REDIS_URL environment variable is not set")

celery = Celery(
    "application",
    broker=redis_url,
    backend=redis_url
)

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)

parsed_url = urlparse(redis_url)
if is_production and parsed_url.scheme == "rediss":
    celery.conf.update(
        redis_backend_use_ssl={
            "ssl_cert_reqs": ssl.CERT_REQUIRED
        }
    )

from application.tasks import stream_task
