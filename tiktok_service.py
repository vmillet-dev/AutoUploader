import logging
import time

import undetected_chromedriver as uc

class TiktokService:
    logger = logging.getLogger("TiktokService")
    driver = None

    def __init__(self):
        self.driver = uc.Chrome()

    def get_videos_by_tag(self, tag):
        self.driver.get(f"https://tiktok.com/tag/{tag}")
        time.sleep(10)
