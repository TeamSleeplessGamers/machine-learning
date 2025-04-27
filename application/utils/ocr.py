import cv2
import pytesseract
import numpy as np

def preprocess_for_numbers(frame):
    """Preprocess the frame to improve number detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh

def extract_first_number(frame):
    """Extract only the first detected number from the frame."""
    processed = preprocess_for_numbers(frame)

    # OCR config to detect ONLY digits
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(processed, config=custom_config)

    # Split into tokens and find first pure number
    tokens = text.split()
    for token in tokens:
        if token.isdigit():
            return int(token)

    return None  # If no valid number found

def detect_number_from_frame(frame):
    """Main entry point."""
    number = extract_first_number(frame)
    return number