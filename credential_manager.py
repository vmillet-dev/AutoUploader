import json
import logging
import os
from datetime import datetime

class CredentialManager:
    __TOKEN_FILE = '/token.json'
    __CLIENT_SECRETS_FILE = '/client_secret.json'
    __CREDENTIAL_TRACKER_FILE = 'credential_tracker.json'

    logger = logging.getLogger("CredentialManager")
    client_secret_data = None
    token_data = None

    def __init__(self, base_auth_folder):
        tracker = self.__load_credential_tracker()
        if len(tracker['credentials']) == 0:
            self.__init_tracker(base_auth_folder, tracker)
        self.__next_credential_available()

    def update_current_credential(self, quota_exhausted=False):
        self.logger.info("Updating current credential")
        tracker = self.__load_credential_tracker()
        current_credential = tracker['credentials'][tracker['current_index']]
        current_credential['last_used'] = datetime.now().isoformat()
        if quota_exhausted:
            current_credential['quota_exhausted'] = True
            self.__save_credential_tracker(tracker)
            if self.__next_credential_available() is not True:
                raise Exception("No more credentials available")

    def reset_all_credential_tracker(self):
        tracker = self.__load_credential_tracker()
        for cred in tracker['credentials']:
            cred['quota_exhausted'] = False
        tracker['current_index'] = 0
        self.__save_credential_tracker(tracker)

    def __init_tracker(self, base_auth_folder, tracker):
        self.logger.info("Init tracker file")
        credential_folders = os.listdir(base_auth_folder)
        if len(credential_folders) == 0:
            raise Exception("No credential folders found")
        for credential in credential_folders:
            tracker['credentials'].append({
                "folder": base_auth_folder + credential,
                "last_used": None,
                "quota_exhausted": False
            })
        self.__save_credential_tracker(tracker)

    def __next_credential_available(self):
        tracker = self.__load_credential_tracker()
        next_index = tracker['current_index']
        while True:
            if next_index < len(tracker['credentials']):
                if tracker['credentials'][next_index]['quota_exhausted'] is not True:
                    tracker['current_index'] = next_index
                    self.__save_credential_tracker(tracker)
                    self.__load_current_credential(tracker)
                    self.logger.info(f"Next credential available found, loading {tracker['credentials'][next_index]['folder']}")
                    return True
                else:
                    self.logger.info(f"Credential at {tracker['credentials'][next_index]['folder']} exceeds quota, trying next one...")
                    next_index += 1
            else:
                next_index = 0
            if next_index == tracker['current_index']:
                self.client_secret_data = None
                self.token_data = None
                self.logger.warning("No more credentials available")
                return False

    def __load_credential_tracker(self):
        if os.path.exists(self.__CREDENTIAL_TRACKER_FILE):
            with open(self.__CREDENTIAL_TRACKER_FILE, 'r') as f:
                return json.load(f)
        self.logger.warning(f"{self.__CREDENTIAL_TRACKER_FILE} not found. Creating new one")
        return {"credentials": [], "current_index": 0}

    def __save_credential_tracker(self, tracker):
        with open(self.__CREDENTIAL_TRACKER_FILE, 'w') as f:
            json.dump(tracker, f, indent=2)

    def __load_data_from_json(self, tracker, file):
        path = tracker['credentials'][tracker['current_index']]['folder'] + file
        if os.path.exists(path):
            try:
                with open(path, 'r') as token_file:
                    return json.load(token_file)
            except Exception as e:
                self.logger.error(f"Error loading token file: {str(e)}")

    def __load_current_credential(self, tracker):
        self.client_secret_data = self.__load_data_from_json(tracker, self.__CLIENT_SECRETS_FILE)
        self.token_data = self.__load_data_from_json(tracker, self.__TOKEN_FILE)
        self.logger.info(f"Loaded existing credentials from token file {self.__TOKEN_FILE} and {self.__CLIENT_SECRETS_FILE}")
