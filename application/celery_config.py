from celery import Celery

celery_app = Celery('tasks', broker='rediss://default:AVNS_mYGDB90UM2d26EpCQYB@db-redis-nyc3-68847-do-user-15873533-0.c.db.ondigitalocean.com:25061/1')
