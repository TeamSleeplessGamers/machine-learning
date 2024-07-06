from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode
chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

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
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    finally:
        driver.quit()
