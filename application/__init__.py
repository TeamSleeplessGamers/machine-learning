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
    except Exception as e:
        app.logger.error(f"Error initializing Firebase: {e}")

    CORS(app, origins='*')

    app.register_blueprint(routes_bp)

    return app
