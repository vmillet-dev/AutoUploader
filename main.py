import logging

from tiktok_service import TiktokService

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    service = TiktokService()
    service.get_videos_by_tag("test")

