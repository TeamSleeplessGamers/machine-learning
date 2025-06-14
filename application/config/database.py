import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date, datetime

class Database:
    def __init__(self):
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        try:
            self.database_url = os.environ.get('DATABASE_URL')
            self.conn = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor
            )
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
    def get_user_by_twitch_channel(self, twitch_profile):
        if not self.conn:
            print("No database connection.")
            return []

        try:
            with self.conn.cursor() as cursor:
                query = "SELECT * FROM users WHERE twitch_profile IS NOT NULL AND LOWER(twitch_profile) = LOWER(%s)"
                cursor.execute(query, (twitch_profile,))
                user = cursor.fetchone()
                return user
        except Exception as e:
            print(f"Error executing SELECT query: {e}")
            return {}
    def get_match_duration_by_event_id(self, event_id):
        if not self.conn:
            print("No database connection.")
            return None
        try:
            with self.conn.cursor() as cursor:
                query = "SELECT time_limit FROM events WHERE id = %s"
                cursor.execute(query, (event_id,))
                result = cursor.fetchone()
                time_limit = result['time_limit']
                return time_limit if result else None
        except Exception as e:
            print(f"Error executing SELECT query: {e}")
            return None
        
    def get_match_events_for_today(self):
        if not self.conn:
            print("No database connection.")
            return []
        try:
            today = date.today()
            with self.conn.cursor() as cursor:
                query = """
                    SELECT id, start_date, time_limit, emails
                    FROM events
                    WHERE DATE(start_date) = %s
                """
                cursor.execute(query, (today,))
                results = cursor.fetchall()
                return results if results else []
        except Exception as e:
            print(f"Error executing SELECT query: {e}")
            return []

    def update_gpu_task_for_event_start_soon(self, entity_type, entity_id, pod_id, status, stopped_at=None):
        if not self.conn:
            print("No database connection.")
            return []
        started_at = datetime.utcnow()
        
        try:
            with self.conn.cursor() as cursor:
                query = """
                INSERT INTO gpu_task_runs (
                    entity_type, entity_id, pod_id, started_at, stopped_at, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (entity_type, entity_id, pod_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    stopped_at = EXCLUDED.stopped_at,
                    updated_at = CURRENT_TIMESTAMP;
            """
                cursor.execute(query, (entity_type, entity_id, pod_id, started_at, stopped_at, status))
                self.conn.commit()
                print(f"GPU task updated for {entity_type} ID {entity_id} with status {status}.")
        except Exception as e:
            print(f"Error updating GPU task for {entity_type} ID {entity_id}: {e}")

    def is_gpu_task_running_for_entity(self, entity_type, entity_id):
        if not self.conn:
            print("No database connection.")
            return False
        try:
            with self.conn.cursor() as cursor:
                query = """
                SELECT 1 FROM gpu_task_runs
                WHERE entity_type = %s AND entity_id = %s AND status IN ('running')
                LIMIT 1
                """
                cursor.execute(query, (entity_type, entity_id))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking GPU task status for {entity_type} ID {entity_id}: {e}")
            return False

    def get_gpu_pod_ids_for_entity(self, entity_type, entity_id):
        if not self.conn:
            print("No database connection Get.")
            return []
        try:
            with self.conn.cursor() as cursor:
                query = """
                SELECT pod_id FROM gpu_task_runs
                WHERE entity_type = %s AND entity_id = %s AND status = 'running'
                """
                cursor.execute(query, (entity_type, entity_id))
                results = cursor.fetchall()
                
                # results is a list of dicts (or tuples depending on cursor config)
                pod_ids = [row['pod_id'] for row in results]
                return pod_ids
        except Exception as e:
            print(f"Error fetching GPU pod IDs for {entity_type} ID {entity_id}: {e}")
            return []