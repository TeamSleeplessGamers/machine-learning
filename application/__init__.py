from flask import Flask
from flask_cors import CORS
from .config.firebase import initialize_firebase
from .routes.routes import routes_bp
import cv2

def create_app():
    """Factory to create the Flask application.
    
    :return: A `Flask` application instance.
    """
    app = Flask(__name__)

    try:
        # initialize_firebase()
        image_path = '/Users/trell/Projects/machine-learning/frames/output_gray_frame_2400.jpg'
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Failed to load image at {image_path}")

        # Convert the image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.threshold(gray_frame, 0, 255,
	        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh_frame, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

        output = frame.copy()
        componentMask = (labels == i).astype("uint8") * 255
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        cv2.imwrite('/Users/trell/Projects/machine-learning/frames_processed/processed_gray_frame_2400.jpg', componentMask)
    except Exception as e:
        app.logger.error(f"Error initializing Firebase: {e}")

    CORS(app, origins='*')

    app.register_blueprint(routes_bp)

    return app
