import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode
chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration
chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
chrome_options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# URL of the website to scrape
url = 'https://www.sleeplessgamers.com/event/?id=6'

# Function to fetch and process the webpage
def fetch_and_process():
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

        print("Twitch URLs:", twitch_urls)

        # Further processing can be done here

    except Exception as e:
        print(f"Error fetching page content: {e}")

# Main loop to continuously scrape the website
def main():
    interval = 60  # Interval in seconds between each fetch
    while True:
        fetch_and_process()
        time.sleep(interval)

if __name__ == "__main__":
    try:
        main()
    finally:
        driver.quit()
