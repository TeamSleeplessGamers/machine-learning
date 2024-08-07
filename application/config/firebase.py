# firebase.py
import firebase_admin
from firebase_admin import credentials, db
import os

def initialize_firebase():
    # Initialize Firebase
    if not firebase_admin._apps:
        print("lets start", os.getenv('FIREBASE_CRED_PATH'))
        cred = credentials.Certificate(os.getenv('FIREBASE_CRED_PATH'))
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
        })

def get_database():
    return db.reference()
