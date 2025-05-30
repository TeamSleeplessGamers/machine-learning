from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()
celery_app = Celery('tasks', broker= os.environ.get('REDIS_URL'))
