import os
import json
import time
import random
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# File to store credentials
TOKEN_FILE = 'token.json'
CLIENT_SECRETS_FILE = 'client_secret.json'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
CREDENTIAL_TRACKER_FILE = 'credential_tracker.json'
AUTH_BASE_FOLDER = 'youtube_uploader/auth/'

def load_credential_tracker():
    if os.path.exists(CREDENTIAL_TRACKER_FILE):
        with open(CREDENTIAL_TRACKER_FILE, 'r') as f:
            return json.load(f)
    return {"credentials": [], "current_index": 0}

def save_credential_tracker(tracker):
    with open(CREDENTIAL_TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

def update_credential_tracker(auth_folder, used=False, quota_exhausted=False):
    """
    Update the credential tracker with usage information for a specific auth folder.
    This function is crucial for the credential rotation system, keeping track of which
    credentials have been used and which have exhausted their quota.
    """
    tracker = load_credential_tracker()
    for cred in tracker["credentials"]:
        if cred["folder"] == auth_folder:
            if used:
                cred["last_used"] = datetime.now().isoformat()
            cred["quota_exhausted"] = quota_exhausted
            break
    else:
        tracker["credentials"].append({
            "folder": auth_folder,
            "last_used": datetime.now().isoformat() if used else None,
            "quota_exhausted": quota_exhausted
        })
    save_credential_tracker(tracker)

def reset_credential_tracker():
    """
    Reset the quota exhausted status for all credentials and set the current index to 0.
    This function is called after the 12-hour wait period to allow all credentials to be used again.
    """
    tracker = load_credential_tracker()
    for cred in tracker["credentials"]:
        cred["quota_exhausted"] = False
    tracker["current_index"] = 0
    save_credential_tracker(tracker)

def get_next_auth_file():
    tracker = load_credential_tracker()
    available_creds = [cred for cred in tracker["credentials"] if not cred["quota_exhausted"]]
    if not available_creds:
        reset_credential_tracker()
        return get_next_auth_file()
    next_cred = available_creds[tracker["current_index"] % len(available_creds)]
    tracker["current_index"] = (tracker["current_index"] + 1) % len(available_creds)
    save_credential_tracker(tracker)
    return next_cred["folder"]


def get_authenticated_service(auth_folder):
    creds = None
    token_file_path = os.path.join(AUTH_BASE_FOLDER, auth_folder, TOKEN_FILE)
    client_secrets_file_path = os.path.join(AUTH_BASE_FOLDER, auth_folder, CLIENT_SECRETS_FILE)

    print(f"Attempting to authenticate using folder: {auth_folder}")
    print(f"Token file path: {token_file_path}")
    print(f"Client secrets file path: {client_secrets_file_path}")

    # Load credentials from file if they exist
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, 'r') as token_file:
                token_data = json.load(token_file)
                creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            print("Loaded existing credentials from token file")
        except Exception as e:
            print(f"Error loading token file: {str(e)}")

    # If there are no (valid) credentials, attempt to create new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {str(e)}")
        else:
            print("Creating new credentials")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file_path, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Error creating new credentials: {str(e)}")
                raise

        # Save the credentials for future use
        try:
            with open(token_file_path, 'w') as token_file:
                token_file.write(creds.to_json())
            print("Saved new credentials to token file")
        except Exception as e:
            print(f"Error saving new credentials: {str(e)}")

    # Update credential tracker to mark this auth folder as used
    update_credential_tracker(auth_folder, used=True)

    if creds:
        print("Successfully authenticated")
        return build('youtube', 'v3', credentials=creds)
    else:
        print("Failed to authenticate")
        raise Exception("Authentication failed")

def get_authenticated_service_with_retry(max_retries=3):
    for attempt in range(max_retries):
        auth_folder = get_next_auth_file()
        try:
            print(f"Attempt {attempt + 1}/{max_retries} using auth folder: {auth_folder}")
            youtube = get_authenticated_service(auth_folder)
            return youtube
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print(f"Quota exceeded for {auth_folder}. Trying next auth file.")
                # Mark this auth folder as quota exhausted in the credential tracker
                update_credential_tracker(auth_folder, quota_exhausted=True)
            else:
                print(f"Error during authentication: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during authentication: {str(e)}")

    print("All auth files exhausted. Waiting for 12 hours.")
    time.sleep(12 * 60 * 60)  # Wait for 10 seconds in testing mode, otherwise 12 hours
    # Reset the credential tracker after the wait period
    reset_credential_tracker()
    return get_authenticated_service_with_retry()

def upload_video(youtube, file_path, title, description, tags):
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': '22',  # "People & Blogs"
        },
        'status': {
            'privacyStatus': 'private',  # or 'private' or 'unlisted'
            'publishAt': (datetime.now() + timedelta(seconds=random.randint(120, 21600))).isoformat()
        },
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    # Upload the video
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = request.execute()
    print(f"Video uploaded with ID: {response['id']}")
