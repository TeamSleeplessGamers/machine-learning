#!/bin/bash

set -e  # Exit on any error


REPO_URL="https://github.com/TeamSleeplessGamers/machine-learning"
REPO_DIR="machine-learning"

echo "=== Checking if repository is already cloned ==="
if [ -d "$REPO_DIR/.git" ]; then
  echo "Repository already cloned. Skipping clone."
else
  echo "Cloning repository..."
  git clone "$REPO_URL"
fi

cd "$REPO_DIR"

echo "=== Setting up virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing remaining Python dependencies ==="
pip install -r requirements.txt

echo "=== Copying Firebase key file ==="
mkdir -p /workspace
echo "$FIREBASE_KEY_BASE64" | base64 -d > /workspace/firebase-dev.json
cp /workspace/firebase-dev.json firebase-dev.json

echo "=== Waiting for Redis to be reachable at $REDIS_HOST:$REDIS_PORT ==="
until nc -z $REDIS_HOST $REDIS_PORT; do
  echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
  sleep 2
done

echo "=== Starting Celery GPU worker for event ID: $EVENT_ID ==="
exec venv/bin/celery -A application.celery_config.celery worker \
  -Q gpu_tasks_event_"${EVENT_ID}" \
  --loglevel=info \
  --pool=threads \
  --concurrency=8
