import os
from flask import Flask
from flask_cors import CORS
from .config.firebase import initialize_firebase
from dotenv import load_dotenv
from .routes.routes import routes_bp

def create_app():
    load_dotenv()
    app = Flask(__name__)

    try:
        initialize_firebase()
    except Exception as e:
        app.logger.error(f"Error initializing Firebase: {e}")

    CORS(app, origins='*')
    app.register_blueprint(routes_bp)

    return app
