import logging
from pathlib import Path

from config import Config
from video_editor import VideoEditor
from video_source.tiktok import TiktokSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_test_environment():
    """Set up test directories and logging."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('TestVideoEditor')

    # Create test directories if they don't exist
    test_dirs = ['input_videos', 'output_videos']
    for dir_name in test_dirs:
        path = Path(dir_name)
        path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {path}")

def main():
    """Test the VideoEditor class functionality."""
    setup_test_environment()
    logger = logging.getLogger('TestVideoEditor')

    # Initialize VideoEditor with config
    config_path = 'config/config.yaml'
    try:
        editor = VideoEditor(config_path)
        logger.info("Successfully initialized VideoEditor")
    except Exception as e:
        logger.error(f"Failed to initialize VideoEditor: {str(e)}")
        return

    # Check if input directory exists and has videos
    input_dir = Path(editor.config['input_directory'])
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return

    logger.info(f"Found {len(video_files)} video files")

    try:
        # Process videos
        output_path = editor.process_videos()
        logger.info(f"Successfully created video compilation: {output_path}")

        # Verify output file exists and has size > 0
        output_file = Path(output_path)
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info("Verification passed: Output file exists and is not empty")
        else:
            logger.error("Verification failed: Output file is missing or empty")

    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")

if __name__ == '__main__':
    config = Config.load_config('config/config.yaml')

    tiktok = TiktokSource(config)
    tiktok.get_video_by_keyword('fail', 10)
