import logging
import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from app.auth_manager import AuthManager

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

class YoutubeService:

    logger = logging.getLogger("YoutubeService")

    def __init__(self, config):
        self.logger.setLevel(logging.DEBUG)
        self.auth_manager = AuthManager(config)

    def get_credentials(self):
        """Retrieve credentials from the token file or refresh using the client secret file."""
        if os.path.exists(self.auth_manager.get_current_token_json_file()):
            credentials = Credentials.from_authorized_user_file(self.auth_manager.get_current_token_json_file(), SCOPES)
            # Refresh token if needed
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    self.save_credentials(credentials)
                except Exception as e:
                    self.logger.error("Error refreshing token:", e)
                    credentials = self.create_credentials()
            return credentials
        else:
            return self.create_credentials()

    def create_credentials(self):
        """Create new credentials using the client secret file."""
        flow = InstalledAppFlow.from_client_secrets_file(self.auth_manager.get_current_client_secret_file(), SCOPES)
        credentials = flow.run_local_server(port=0)
        self.save_credentials(credentials)
        return credentials

    def save_credentials(self, credentials):
        """Save the credentials to the token file."""
        with open(self.auth_manager.get_current_token_json_file(), 'w') as token:
            token.write(credentials.to_json())

    def upload_video(self, file_path, title, description, category_id=22, privacy_status="public"):
        """Upload a video to YouTube."""
        try:
            youtube = build("youtube", "v3", credentials=self.get_credentials())

            request_body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "categoryId": category_id
                },
                "status": {
                    "privacyStatus": privacy_status
                }
            }

            # Upload the video
            media_body = MediaFileUpload(file_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(
                part="snippet,status",
                body=request_body,
                media_body=media_body
            )
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Uploaded {int(status.progress() * 100)}%...")
            self.auth_manager.update_current_credential()
            print("Upload complete!")
            return response
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                self.auth_manager.update_current_credential(quota_exhausted=True)
                return self.upload_video(file_path, title, description, category_id=22, privacy_status="public")
            return None
        except Exception as e:
            print("An error occurred during upload:", e)
            return None