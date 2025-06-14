#!/bin/bash

echo "=== Installing system dependencies ==="
sudo apt update && sudo apt install -y git python3.13 python3-pip python3-venv netcat

echo "=== Cloning repository ==="
git clone https://github.com/TeamSleeplessGamers/machine-learning
cd machine-learning

echo "=== Setting up virtual environment ==="
python3 -m venv venv

# Activate the venv for the current script execution only
source venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Copying .env file into project directory ==="
cp ../.env .env

echo "=== Loading environment variables ==="
set -a
source .env
set +a

echo "=== Copying firebase json file into project directory ==="
echo "$FIREBASE_KEY_BASE64" | base64 -d > /workspace/firebase-dev.json

cp ../firebase-dev.json firebase-dev.json

echo "=== Waiting for Redis to be reachable at $REDIS_HOST:$REDIS_PORT ==="
until nc -z $REDIS_HOST $REDIS_PORT; do
  echo "Waiting for Redis..."
  sleep 2
done

echo "=== Starting Celery GPU worker ==="
# Use celery from the virtualenv explicitly
venv/bin/celery -A application.celery_config.celery worker -Q gpu_tasks --loglevel=info --pool=threads --concurrency=1
