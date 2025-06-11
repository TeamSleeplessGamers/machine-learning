Machine Learning - Updating Code on DigitalOcean

Steps to Pull the Latest Code into Your VM

Follow these steps to update the machine learning project on your DigitalOcean virtual machine (VM):

1. Log in as the sleepless user

Run the following command to switch to the sleepless user:

su --login sleepless

2. Navigate to the Project Directory

Move to the correct directory where the project is stored:

cd /home/sleepless/environments/machine_learning

3. Pull the Latest Code from Git

Fetch the latest updates from the repository:

git pull

4. (Optional) Restart Services

If your project runs as a service, restart it to apply changes:

sudo systemctl restart gunicorn.service

5. (Optional) Activate Virtual Environment and Install Dependencies

If dependencies were updated, activate the virtual environment and install them:

source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
deactivate  # Exit virtual environment when done

Notes

Ensure you have the necessary permissions to execute these commands.

If you encounter any merge conflicts during git pull, resolve them manually before proceeding.

Always restart services if code updates affect running processes.

# Testing in DO

Run this command in machine_learning project

gunicorn wsgi:app


# Link to documentation

https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uswgi-and-nginx-on-ubuntu-18-04

https://www.digitalocean.com/community/tutorials/install-cuda-cudnn-for-gpu#installing-cuda-on-ubuntu

# Local Development with Celery

To run the Celery worker locally during development, follow these steps:

1. **Activate your virtual environment** (if you use one):

    ```bash
    source venv/bin/activate
    ```

2. **Start the Celery worker:**

    Make sure you run this command from the project root directory (where the `application` folder is located):

    ```bash
    celery -A application.celery_config worker --loglevel=info --pool=threads --concurrency=1
    ```

    - `-A application.celery_config` points Celery to your app instance.
    - `--loglevel=info` will show detailed logs in the console.

3. **Run your Flask app as usual** (e.g., with Flask’s built-in server or Gunicorn).

4. **Sending tasks:**

    Your Flask app (or other parts of your code) can enqueue tasks to Celery as usual, e.g.:

    ```python
    from application.tasks import process_warzone_video_stream_info_task

    process_warzone_video_stream_info_task.delay(username, event_id, user_id, team_id)
    ```

---

# Deployment and Updating on DigitalOcean

1. Log in as the sleepless user:

    ```bash
    su --login sleepless
    ```

2. Navigate to the project directory:

    ```bash
    cd /home/sleepless/environments/machine_learning
    ```

3. Pull the latest code updates from your Git repository:

    ```bash
    git pull
    ```

4. (Optional) Restart Gunicorn to apply changes:

    ```bash
    sudo systemctl restart gunicorn.service
    ```

5. (Optional) Activate the virtual environment and update dependencies if needed:

    ```bash
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate
    ```

6. (Optional) Run Gunicorn manually to test:

    ```bash
    gunicorn wsgi:app
    ```

---

# Useful Links

- [Serving Flask applications with uWSGI and Nginx on Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uswgi-and-nginx-on-ubuntu-18-04)
- [Install CUDA and cuDNN for GPU](https://www.digitalocean.com/community/tutorials/install-cuda-cudnn-for-gpu#installing-cuda-on-ubuntu)
