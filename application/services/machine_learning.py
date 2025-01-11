import requests
import base64
import cv2

# Replace with your Google Cloud Vision API key

def detect_text_with_api_key(roi):
    """
    Detects text in an ROI (Region of Interest) using the Google Cloud Vision API.

    Args:
        roi (np.ndarray): The ROI (image portion) as a NumPy array.

    Returns:
        list: A list of detected text strings.
    """
    api_key = "AIzaSyAAg0O6pGhA3NE3xFN2JV_ZS8KxcNROWkw"

    # Google Cloud Vision API endpoint
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Encode the ROI into a PNG image
    success, encoded_image = cv2.imencode('.png', roi)
    if not success:
        raise Exception("Failed to encode ROI as an image")

    # Base64 encode the image
    content = base64.b64encode(encoded_image).decode("utf-8")

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