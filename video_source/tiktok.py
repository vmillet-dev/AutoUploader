import json
import logging
import random
import time
from logging import DEBUG
import yt_dlp

import undetected_chromedriver as uc
from selenium.webdriver.support.wait import WebDriverWait

from video_source import VideoSource

class TiktokSource(VideoSource):
    logger = logging.getLogger("TiktokSource")
    driver = None

    def __init__(self, config):
        self.config = config
        self.logger.setLevel(DEBUG)
        self.keyword_with_urls = self.__load_fetched_urls()

    def get_video_by_keyword(self, keyword, amount=1):
        try:
            options = uc.ChromeOptions()
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            self.driver = uc.Chrome(options=options)

            self.logger.debug(f"Opening tiktok page with tag {keyword}, searching {amount} video(s)")
            self.driver.get(f"https://tiktok.com/tag/{keyword}")
            self.__wait_for_page_ready()

            time.sleep(2)

            video_urls = self.__extract_video_urls_from_logs(keyword, amount)
            self.logger.info(f"Amount of new videos found: {len(video_urls)}")
            self.__download_video(video_urls)
            self.__save_fetched_urls()
        finally:
            self.driver.quit()

    def __extract_video_urls_from_logs(self, keyword, amount):
        """Efficiently extracts up to `amount` video URLs from network logs."""
        video_urls = []
        has_more_videos = True

        while has_more_videos:
            logs = self.driver.get_log('performance')
            for log_entry in logs:
                # Early filtering of relevant logs
                log_data = json.loads(log_entry['message'])['message']
                if log_data.get('method') != 'Network.responseReceived':
                    continue

                response_url = log_data.get('params', {}).get('response', {}).get('url', '')
                if 'item_list' not in response_url:
                    continue

                # Fetch and parse the response body
                response_body = self.driver.execute_cdp_cmd(
                    'Network.getResponseBody',
                    {'requestId': log_data['params']['requestId']}
                )
                if not response_body or 'body' not in response_body:
                    continue

                # Directly parse relevant fields from the response body
                response_data = json.loads(response_body['body'])
                if isinstance(response_data.get('hasMore'), bool):
                    has_more_videos = response_data.get('hasMore')
                for item in response_data.get('itemList', []):
                    self.__handle_response_item(video_urls, keyword, item)
                    if len(video_urls) >= amount:  # Stop early if we have enough URLs
                        return video_urls
                break
            if has_more_videos:
                self.logger.info('Scrolling to get more videos')
                self.__scroll()
            else:
                self.logger.warning('No more videos on this tag')

        return video_urls

    def __wait_for_page_ready(self, timeout=10):
        WebDriverWait(self.driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        self.logger.debug(f"Page ready")

    def __scroll(self):
        current_position = self.driver.execute_script("return window.pageYOffset;")
        viewport_height = self.driver.execute_script("return window.innerHeight")

        scroll_distance = random.randint(int(viewport_height / 10), int(viewport_height * 0.8))
        target_position = current_position + scroll_distance

        self.driver.execute_script(f"window.scrollTo({{ top: {target_position}, behavior: 'smooth' }});")

        time.sleep(random.uniform(1, 3))

    def __handle_response_item(self, video_urls, keyword, item):
        if self.__is_valid_video(item):
            video_url = self.__construct_video_url(item)
            if (video_url
                and ((
                        keyword in self.keyword_with_urls
                        and any(isinstance(entry, dict) and entry.get("url") != video_url for entry in self.keyword_with_urls.get(keyword, []))
                    )
                    or keyword not in self.keyword_with_urls)
                ):
                self.logger.info(f"New video url found: {video_url}")
                video_urls.append(video_url)
                self.__add_tag_value(keyword, {
                    'url': video_url,
                    'description': item['desc'],
                    'create_time': item['createTime']
                })

    @staticmethod
    def __construct_video_url(item):
        """Constructs a video URL from item data if valid."""
        author = item.get('author', {})
        video_id = item.get('id')
        nickname = author.get('uniqueId')

        if nickname and video_id:
            return f"https://www.tiktok.com/@{nickname}/video/{video_id}"
        return None

    @staticmethod
    def __is_valid_video(item) -> bool:
        return item['officalItem'] is not True and item['itemMute'] is not True

    def __load_fetched_urls(self):
        """Load previously fetched URLs from a JSON file as a dictionary of lists."""
        try:
            with open(self.config.input_output.video_urls_by_keyword_file, "r") as file:
                data = json.load(file)
                # Ensure data is in the expected dictionary format
                if isinstance(data, dict):
                    return data
                else:
                    return {}
        except (FileNotFoundError, json.JSONDecodeError):
            # Return an empty dictionary if the file doesn't exist or is not valid JSON
            return {}

    def __save_fetched_urls(self):
        """Save fetched URLs to a JSON file directly as lists."""
        with open(self.config.input_output.video_urls_by_keyword_file, "w") as file:
            json.dump(self.keyword_with_urls, file, indent=4)

    def __add_tag_value(self, tag, value):
        """Adds a URL to the tag's list in the dictionary."""
        if tag not in self.keyword_with_urls:
            self.keyword_with_urls[tag] = []
        self.keyword_with_urls[tag].append(value)

    def __download_video(self, urls):
        ydl_opts = {
            'format': 'best',  # Download best quality
            'outtmpl': str(self.config.input_output.input_videos_dir + 'video_%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'logger': self.logger,
            'writesubtitles': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(urls)
            self.logger.info(f'Successfully downloaded {len(urls)} video(s)')

        except Exception as e:
            self.logger.error(f'Failed to download videos: {str(e)}')

