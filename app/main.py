import argparse
import logging
from pathlib import Path
import sys
import time

from app.config import Config
from video_editor import VideoEditor
from video_source.tiktok import TiktokSource

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cleanup_temp_files(input_dir):
    """Remove temporary video files."""
    for file in Path(input_dir).glob('*.mp4'):
        file.unlink()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download TikTok videos by tag and create compilation')
    parser.add_argument('tag', help='TikTok tag to search for')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files after processing')
    args = parser.parse_args()

    # Initialize logging and configuration
    logger = logging.getLogger('AutoUploader')
    logger.info(f"Starting video compilation for tag: {args.tag}")
    config = Config.load_config(args.config)

    # Download videos using TiktokSource
    tiktok = TiktokSource(config)
    amount = getattr(config.video_selection, 'amount', 5)
    tiktok.get_video_by_keyword(args.tag, amount)
    time.sleep(2)  # Wait for downloads to complete

    # Create compilation using VideoEditor
    editor = VideoEditor(config)
    output_path = editor.process_videos()
    if not output_path or not Path(output_path).exists():
        logger.error("Failed to create video compilation")
        sys.exit(1)

    # Report success and cleanup
    logger.info(f"Successfully created compilation: {output_path}")
    if not args.keep_temp:
        cleanup_temp_files(config.input_output.input_videos_dir)

if __name__ == '__main__':
    setup_logging()
    main()
