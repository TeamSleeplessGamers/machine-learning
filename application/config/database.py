import os
import psycopg2

class Database:
    def __init__(self):
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        try:
            self.database_url = os.environ['DATABASE_URL']
            self.conn = psycopg2.connect(self.database_url)
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.conn = None

    def get_connection(self):
        return self.conn

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
    def get_list_of_sg_users(self):
        if not self.conn:
            print("No database connection.")
            return []

        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT twitch_profile FROM users WHERE twitch_profile IS NOT NULL AND twitch_profile <> ''")
                users = cursor.fetchall()
                user_list = [user[0] for user in users]
                return list(set(user_list))
        except Exception as e:
            print(f"Error executing SELECT query: {e}")
            return []
