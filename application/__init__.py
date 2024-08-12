from flask import Flask
from flask_cors import CORS
from .config.firebase import initialize_firebase
import pytesseract

from .routes.routes import routes_bp
import cv2

def create_app():
    """Factory to create the Flask application.
    
    :return: A `Flask` application instance.
    """
    app = Flask(__name__)

    try:
        initialize_firebase()
        image_path = '/Users/trell/Projects/machine-learning/frames/output_gray_frame_1200.jpg'
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Failed to load image at {image_path}")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray_frame, 0, 255,
	        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        color_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)
        
        rectangles = [
            (1798, 50, 60, 30),
            (25, 300, 50,50),
        ]
        output_dir = '/Users/trell/Projects/machine-learning/frames_processed'

        for i, (x, y, w, h) in enumerate(rectangles):
            custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789IO'
            roi_frame = color_frame[y:y + h, x:x + w]

            scale_factor = 2
            rescaled_roi = cv2.resize(roi_frame, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LINEAR)

            rgb_roi_frame = cv2.cvtColor(rescaled_roi, cv2.COLOR_BGR2RGB)
            output_path = f'{output_dir}/extracted_region_{i + 1}.jpg'
            cv2.imwrite(output_path, rgb_roi_frame)

            extracted_text = pytesseract.image_to_string(rgb_roi_frame, config=custom_config)
            print(f"Extracted text from region {i + 1}: {extracted_text.strip()}")

        for (x, y, w, h) in rectangles:
            img_with_box = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('/Users/trell/Projects/machine-learning/frames_processed/processed_gray_frame_2400.jpg', img_with_box)
    except Exception as e:
        app.logger.error(f"Error initializing Firebase: {e}")

    CORS(app, origins='*')

    app.register_blueprint(routes_bp)

    return app
