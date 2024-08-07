from dotenv import load_dotenv

load_dotenv(".env")

from application import create_app, app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8000)