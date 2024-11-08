import ffmpeg
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from config import Config

class VideoEditor:
    """A class for automated video editing using ffmpeg."""

    def __init__(self, config: Config):
        """Initialize the VideoEditor with configuration.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = self._setup_logger()

        # Set up input/output paths from nested config
        self.input_dir = Path(self.config.input_output.input_videos_dir)
        self.output_dir = Path(self.config.input_output.output_videos_dir)

        # Create necessary directories
        self.temp_dir = self.output_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('VideoEditor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _get_video_info(self, video_path: str, max_retries: int = 3) -> Dict:
        """Get video metadata using ffprobe."""
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Probing video {video_path} (attempt {attempt + 1}/{max_retries})")
                probe = ffmpeg.probe(video_path)
                if 'streams' not in probe:
                    raise ValueError("No streams found in video file")
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                video_info['streams'] = probe['streams']
                return video_info
            except (ffmpeg.Error, ValueError) as e:
                self.logger.warning(f"FFprobe error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to probe video after {max_retries} attempts: {video_path}")

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe['format']['duration'])
        except ffmpeg.Error as e:
            self.logger.error(f"Error getting duration for {video_path}: {str(e)}")
            raise

    def _is_vertical(self, video_info: Dict) -> bool:
        """Check if video is vertical based on aspect ratio."""
        width = int(video_info.get('width', 0))
        height = int(video_info.get('height', 0))
        return height > width

    def _get_input_videos(self) -> List[str]:
        """Get list of video files from input directory."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(f'*{ext}')))

        if not video_files:
            self.logger.warning(f"No video files found in {self.input_dir}")
            return []

        return [str(f) for f in video_files]

    def _filter_videos_by_duration(self, video_paths: List[str]) -> List[str]:
        """Filter videos based on duration limits from config."""
        min_length = self.config.video_editor.video_duration_limits.min_length
        max_length = self.config.video_editor.video_duration_limits.max_length

        filtered_videos = []
        for path in video_paths:
            try:
                duration = self._get_video_duration(path)
                if min_length <= duration <= max_length:
                    filtered_videos.append(path)
                else:
                    self.logger.info(f"Skipping {path}: Duration {duration:.1f}s outside limits")
            except Exception as e:
                self.logger.error(f"Error processing {path}: {str(e)}")
                continue

        return filtered_videos

    def _prepare_video(self, input_video: str, video_info: Dict) -> Tuple[ffmpeg.Stream, ffmpeg.Stream]:
        """Handle video processing according to config settings."""
        try:
            self.logger.debug(f"Preparing video: {input_video}")
            input_stream = ffmpeg.input(input_video)
            video_stream = input_stream.video
            audio_stream = None

            # Check for audio stream
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    self.logger.debug("Found audio stream")
                    audio_stream = input_stream.audio
                    break

            # Calculate dimensions
            input_width = int(video_info.get('width', 0))
            input_height = int(video_info.get('height', 0))
            target_width = self.config.video_editor.output_resolution.width
            target_height = self.config.video_editor.output_resolution.height
            is_input_vertical = self._is_vertical(video_info)
            is_output_vertical = self.config.video_editor.video_orientation.output_mode == "vertical"

            if input_width == 0 or input_height == 0:
                raise ValueError("Invalid input dimensions")

            self.logger.debug(f"Input dimensions: {input_width}x{input_height}")
            self.logger.debug(f"Target dimensions: {target_width}x{target_height}")
            self.logger.debug(f"Input orientation: {'vertical' if is_input_vertical else 'landscape'}")
            self.logger.debug(f"Output orientation: {self.config.video_editor.video_orientation.output_mode}")

            # Calculate scaling to fit within target dimensions while maintaining aspect ratio
            input_aspect = input_width / input_height
            target_aspect = target_width / target_height

            if is_output_vertical:
                # For vertical output, scale based on width first
                new_width = target_width
                new_height = int(new_width / input_aspect)
                if new_height > target_height:
                    new_height = target_height
                    new_width = int(new_height * input_aspect)
            else:
                # For landscape output, scale based on height first
                new_height = target_height
                new_width = int(new_height * input_aspect)
                if new_width > target_width:
                    new_width = target_width
                    new_height = int(new_width / input_aspect)

            # Ensure dimensions are even numbers
            new_width = new_width if new_width % 2 == 0 else new_width - 1
            new_height = new_height if new_height % 2 == 0 else new_height - 1

            if new_width < 2 or new_height < 2:
                raise ValueError(f"Invalid scaled dimensions: {new_width}x{new_height}")

            self.logger.debug(f"Scaling to: {new_width}x{new_height}")

            # Scale the video
            scaled_video = video_stream.filter('scale', str(new_width), str(new_height))

            # Calculate padding for centered placement
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            # Handle background filling based on orientation mismatch
            if is_input_vertical != is_output_vertical:
                background_type = self.config.video_editor.video_orientation.vertical_video.background_type
                if background_type == "blur":
                    # Create blurred background from input
                    blur_amount = self.config.video_editor.video_orientation.vertical_video.blur_amount
                    background = (
                        video_stream
                        .filter('scale', str(target_width), str(target_height))
                        .filter('boxblur', str(blur_amount))
                    )
                elif background_type in ["image", "video"]:
                    # Use provided background
                    bg_path = self.config.video_editor.video_orientation.vertical_video.background_path
                    if not bg_path:
                        raise ValueError(f"Background path not provided for type: {background_type}")
                    background = (
                        ffmpeg.input(bg_path)
                        .filter('scale', str(target_width), str(target_height))
                    )
                else:
                    # Default to black background
                    background = ffmpeg.input(f'color=c=black:s={target_width}x{target_height}', f='lavfi')
            else:
                # If orientations match, use black background
                background = ffmpeg.input(f'color=c=black:s={target_width}x{target_height}', f='lavfi')

            # Create final video with background and centered content
            processed_video = (
                background
                .overlay(scaled_video, x=x_offset, y=y_offset)
                .filter('format', 'yuv420p')
            )

            return processed_video, audio_stream

        except Exception as e:
            self.logger.error(f"Error preparing video {input_video}: {str(e)}")
            raise

    def _normalize_audio(self, audio_stream: Optional[ffmpeg.Stream]) -> Optional[ffmpeg.Stream]:
        """Apply audio normalization according to config."""
        if audio_stream is None:
            return None

        try:
            if not self.config.video_editor.audio.normalize:
                return audio_stream

            normalized = (
                audio_stream
                .filter('loudnorm', i='-14', lra='7', tp='-1')
                .filter('aformat', sample_rates='44100', sample_fmts='fltp', channel_layouts='stereo')
            )
            return normalized
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return audio_stream

    def _apply_transition(self, video1: str, video2: str, output_path: str):
        """Apply transition effect between two videos."""
        try:
            duration = self.config.video_editor.transitions.duration

            stream1 = ffmpeg.input(video1)
            stream2 = ffmpeg.input(video2)

            # Apply crossfade transition
            stream = ffmpeg.filter(
                [stream1, stream2],
                'xfade',
                transition='fade',
                duration=duration,
                offset=self._get_video_duration(video1) - duration
            )

            # Output with consistent encoding
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec='aac',
                vcodec='libx264',
                preset='medium',
                pix_fmt='yuv420p'
            ).overwrite_output()

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        except Exception as e:
            self.logger.error(f"Error applying transition: {str(e)}")
            raise

    def process_videos(self) -> Optional[str]:
        """Process videos from input directory and create compilation."""
        try:
            # Get and filter input videos
            input_videos = self._get_input_videos()
            if not input_videos:
                raise ValueError("No input videos found")

            filtered_videos = self._filter_videos_by_duration(input_videos)
            if not filtered_videos:
                raise ValueError("No videos meet duration criteria")

            self.logger.info(f"Processing {len(filtered_videos)} videos")

            # Process each video
            processed_videos = []
            for i, video_path in enumerate(filtered_videos):
                try:
                    # Get video info
                    video_info = self._get_video_info(video_path)
                    self.logger.debug(f"Processing video {i+1}/{len(filtered_videos)}: {video_path}")
                    self.logger.debug(f"Video info: {video_info}")

                    # Output processed video
                    output_path = self.temp_dir / f"processed_{i}.mp4"
                    self.logger.info(f"Processing to temporary file: {output_path}")

                    # Build FFmpeg command for video processing
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-vf', f'scale={self.config.video_editor.output_resolution.width}:{self.config.video_editor.output_resolution.height}:force_original_aspect_ratio=decrease,pad={self.config.video_editor.output_resolution.width}:{self.config.video_editor.output_resolution.height}:(ow-iw)/2:(oh-ih)/2',
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac',
                        '-movflags', '+faststart',
                        '-y',
                        str(output_path)
                    ]

                    # Run FFmpeg with subprocess for timeout control
                    try:
                        self.logger.info("Starting FFmpeg processing")
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True
                        )

                        # Monitor progress
                        start_time = time.time()
                        while process.poll() is None:
                            if time.time() - start_time > 300:  # 5 minute timeout
                                process.kill()
                                raise subprocess.TimeoutExpired(cmd, 300)
                            time.sleep(1)

                        stdout, stderr = process.communicate()
                        if process.returncode != 0:
                            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
                        processed_videos.append(str(output_path))
                        self.logger.info(f"Successfully processed video {i+1}/{len(filtered_videos)}")

                    except subprocess.TimeoutExpired:
                        process.kill()
                        self.logger.error("FFmpeg process timed out after 300 seconds")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error during FFmpeg processing: {str(e)}")
                        if process and process.poll() is None:
                            process.kill()
                        continue

                except Exception as e:
                    self.logger.error(f"Error processing video {video_path}: {str(e)}")
                    continue

            if not processed_videos:
                raise ValueError("No videos were successfully processed")

            # Create final compilation
            final_output = self.output_dir / "final_compilation.mp4"
            self.logger.info("Creating final compilation")

            try:
                if len(processed_videos) == 1:
                    self.logger.info("Single video compilation - copying file")
                    shutil.copy2(processed_videos[0], final_output)
                else:
                    # Create concat file
                    concat_file = self.temp_dir / 'concat.txt'
                    with open(concat_file, 'w') as f:
                        for video in processed_videos:
                            f.write(f"file '{Path(video).absolute()}'\n")

                    # Concatenate videos using direct FFmpeg command
                    self.logger.info("Starting video concatenation")
                    cmd = [
                        'ffmpeg',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', str(concat_file),
                        '-c', 'copy',
                        '-movflags', '+faststart',
                        '-y',
                        str(final_output)
                    ]

                    try:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True
                        )

                        # Monitor progress
                        start_time = time.time()
                        while process.poll() is None:
                            if time.time() - start_time > 300:  # 5 minute timeout
                                process.kill()
                                raise subprocess.TimeoutExpired(cmd, 300)
                            time.sleep(1)

                        stdout, stderr = process.communicate()
                        if process.returncode != 0:
                            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
                        self.logger.info("Video concatenation completed successfully")

                    except subprocess.TimeoutExpired:
                        process.kill()
                        self.logger.error("Concatenation process timed out after 300 seconds")
                        raise
                    except Exception as e:
                        self.logger.error(f"Error during concatenation: {str(e)}")
                        if process and process.poll() is None:
                            process.kill()
                        raise

                self.logger.info(f"Successfully created video compilation: {final_output}")
                return str(final_output)
            except Exception as e:
                self.logger.error(f"Error during final compilation: {str(e)}")
                if final_output.exists():
                    final_output.unlink()
                return None

        except Exception as e:
            self.logger.error(f"Error during video compilation: {str(e)}")
            return None

        finally:
            # Clean up temporary files
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir.mkdir(exist_ok=True)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary files: {str(e)}")
