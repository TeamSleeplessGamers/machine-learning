#!/bin/bash

echo "=== Installing system dependencies ==="
sudo apt update && sudo apt install -y git python3.13 python3-pip python3-venv


echo "=== Cloning repository ==="
git clone https://github.com/TeamSleeplessGamers/machine-learning
cd machine-learning

echo "=== Setting up virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Copying .env file into project directory ==="
cp ../.env .env

echo "=== Loading environment variables ==="
source .env

echo "=== Copying firebase json file into project directory ==="
cp ../firebase-dev.json firebase-dev.json

echo "=== Waiting for Redis to be reachable at $REDIS_HOST:$REDIS_PORT ==="
until nc -z $REDIS_HOST $REDIS_PORT; do
  echo "Waiting for Redis..."
  sleep 2
done

# Run this in your shell session
export $(cat .env | xargs)

echo "=== Starting Celery GPU worker ==="
celery -A application.celery_config.celery worker -Q gpu_tasks --loglevel=info --pool=threads --concurrency=1
