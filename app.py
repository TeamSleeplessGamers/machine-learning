from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS
from dotenv import load_dotenv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import psycopg2

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app, origins='*')  # Enable CORS for all origins
app.secret_key = os.urandom(24)

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = 'http://localhost:5001/callback'
token_url = 'https://streamlabs.com/api/v2.0/token'
oauth_url = 'https://streamlabs.com/api/v2.0/authorize'
database_url = os.getenv('DATABASE_URL')

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode
chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Global database connection
conn = None

def initialize_database():
    global conn
    conn = psycopg2.connect(database_url)

@app.route('/')
def index():
    return '''
        <h1>Welcome to Stream Integration</h1>
        <a href="/authorize_streamlabs">Authorize with Streamlabs</a>
    '''

@app.route('/authorize_streamlabs')
def authorize_streamlabs():
    userid = request.args.get('userid') or ""
    auth_url = f'{oauth_url}?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=mediashare.control&state={userid}'
    return jsonify({'auth_url': auth_url})

@app.route('/callback')
def callback():
    code = request.args.get('code')
    userid = request.args.get('state')
    response = requests.post(token_url, data={
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'redirect_uri': redirect_uri
    })

    token_data = response.json()
    session['streamlabs_token'] = token_data['access_token']
    
    # Insert the token into the database with userid
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO user_streamlabs_token (userid, token) VALUES (%s, %s)", (userid, token_data['access_token']))
            conn.commit()
    except Exception as e:
        print(f"Error inserting token into database: {e}")
        return jsonify({'error': 'Failed to insert token into database'}), 500

    return 'Streamlabs authorization successful!'
@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # Open the webpage
        driver.get(url)

        # Wait for the page to load and JavaScript to execute
        time.sleep(5)  # Adjust this delay as needed

        # Find all iframe elements
        iframe_elements = driver.find_elements(By.TAG_NAME, 'iframe')

        # Extract src attribute values from iframe elements
        twitch_urls = []
        for iframe in iframe_elements:
            src = iframe.get_attribute('src')
            if src.startswith('https://player.twitch.tv'):
                twitch_urls.append(src)

        return jsonify({'twitch_urls': twitch_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'

if __name__ == "__main__":
    initialize_database()
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    finally:
        driver.quit()
        if conn:
            conn.close()

