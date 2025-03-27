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

sudo systemctl restart machine_learning

5. (Optional) Activate Virtual Environment and Install Dependencies

If dependencies were updated, activate the virtual environment and install them:

source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
deactivate  # Exit virtual environment when done

Notes

Ensure you have the necessary permissions to execute these commands.

If you encounter any merge conflicts during git pull, resolve them manually before proceeding.

Always restart services if code updates affect running processes.


# Link to documentation

https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uswgi-and-nginx-on-ubuntu-18-04