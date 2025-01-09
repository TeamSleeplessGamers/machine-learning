import requests
import base64
import os

# Replace with your Google Cloud Vision API key

def detect_text_with_api_key(image_path):
    """
    Detects text in an image using the Google Cloud Vision API and API Key.

    Args:
        image_path (str): Path to the image file.
        api_key (str): Your Google Cloud Vision API Key.

    Returns:
        list: A list of detected text strings.
    """
    api_key = "AIzaSyAAg0O6pGhA3NE3xFN2JV_ZS8KxcNROWkw"

    # Google Cloud Vision API endpoint
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        content = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare the request payload
    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    # Send the POST request
    response = requests.post(url, json=payload)

    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.text}")

    # Parse and return detected text
    response_data = response.json()
    if "textAnnotations" in response_data["responses"][0]:
        detected_texts = [
            annotation["description"]
            for annotation in response_data["responses"][0]["textAnnotations"]
        ]
        return detected_texts
    else:
        return ["No text detected."]