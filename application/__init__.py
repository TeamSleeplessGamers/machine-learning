from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from .config.firebase import initialize_firebase
from .routes.routes import routes_bp

app = Flask(__name__)

def create_app():
    """Factory to create the Flask application
    :return: A `Flask` application instance
    """
    load_dotenv(".env")
    initialize_firebase()
    CORS(app, origins='*')

    # Register the Blueprint
    app.register_blueprint(routes_bp)

    return app