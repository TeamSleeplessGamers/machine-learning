# firebase.py
import firebase_admin
from firebase_admin import credentials, db
import os

def initialize_firebase():
    # Initialize Firebase
    if not firebase_admin._apps:
        cred  = credentials.Certificate(os.environ['FIREBASE_CRED_PATH'])
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.environ['FIREBASE_DATABASE_URL']
        })
def get_database():
    return db.reference()
