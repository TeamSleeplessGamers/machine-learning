import subprocess

class TwitchRecorder:
    def __init__(self, username, event_id, user_id, team_id):
        self.username = username
        self.event_id = event_id
        self.user_id = user_id
        self.team_id = team_id

    def get_live_stream_url(self):
        try:
            result = subprocess.run(
                [
                    "streamlink", 
                    "--twitch-disable-ads", 
                    f"https://twitch.tv/{self.username}", 
                    "best", 
                    "--stream-url"
                ],
                check=True,
                text=True,
                capture_output=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"[Streamlink Error] {e.stderr}")
            return None