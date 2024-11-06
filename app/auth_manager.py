import json
import logging
import os
from datetime import datetime
from logging import DEBUG


class AuthManager:
    logger = logging.getLogger("AuthManager")

    def __init__(self, config):
        self.logger.setLevel(DEBUG)
        self.token_file_path = config.input_output.token_file_name
        self.client_secret_file_path = config.input_output.client_secret_file_name
        self.credential_tracker_file = config.input_output.credential_tracker_file_name
        tracker = self.__load_credential_tracker()
        if len(tracker['credentials']) == 0:
            self.__init_tracker(config.input_output.auth_dir, tracker)
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

    def save_refrsh_token(self, data):
        tracker = self.__load_credential_tracker()
        with open(tracker['credentials'][tracker['current_index']]['folder'] + self.token_file_path, 'w') as token_file:
            token_file.write(data)

    def __init_tracker(self, base_auth_folder, tracker):
        self.logger.info("Init tracker file")
        credential_folders = os.listdir(base_auth_folder)
        if len(credential_folders) == 0:
            raise Exception("No credential folders found")
        for credential in credential_folders:
            tracker['credentials'].append({
                "folder": base_auth_folder + credential + '/',
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
                    self.logger.info(f"Credential available found, loading {tracker['credentials'][next_index]['folder']}")
                    return True
                else:
                    self.logger.info(f"Credential at {tracker['credentials'][next_index]['folder']} exceeds quota, trying next one...")
                    next_index += 1
            else:
                next_index = 0
            if next_index == tracker['current_index']:
                tracker['current_index'] = 0
                self.logger.warning("No more credentials available")
                return False

    def __load_credential_tracker(self):
        if os.path.exists(self.credential_tracker_file):
            with open(self.credential_tracker_file, 'r') as f:
                return json.load(f)
        self.logger.warning(f"{self.credential_tracker_file} not found. Creating new one")
        return {"credentials": [], "current_index": 0}

    def __save_credential_tracker(self, tracker):
        with open(self.credential_tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)

    def get_current_client_secret_file(self):
        tracker = self.__load_credential_tracker()
        return tracker['credentials'][tracker['current_index']]['folder'] + self.client_secret_file_path

    def get_current_token_json_file(self):
        tracker = self.__load_credential_tracker()
        return tracker['credentials'][tracker['current_index']]['folder'] + self.token_file_path

