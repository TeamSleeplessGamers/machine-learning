from flask import Flask
from flask_cors import CORS
from .config.firebase import initialize_firebase
from dotenv import load_dotenv
from celery import Celery

from .routes.routes import routes_bp

# Define the Celery instance globally
celery = Celery(__name__, broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

def create_app():
    """Factory to create the Flask application."""
    load_dotenv()
    app = Flask(__name__)

    # If you want to sync config from Flask to Celery:
    app.config.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0'
    )
    celery.conf.update(app.config)

    try:
        initialize_firebase()
    except Exception as e:
        app.logger.error(f"Error initializing Firebase: {e}")

    CORS(app, origins='*')
    app.register_blueprint(routes_bp)

    return app
