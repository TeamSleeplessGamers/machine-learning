from app_factory import create_app
import waitress

app = create_app()

if __name__ == '__main__':
    # Use waitress to serve the app
    waitress.serve(app, host='0.0.0.0', port=8000)