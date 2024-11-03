import ffmpeg
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from config import Config
from video_editor.clip_analyzer import ClipAnalyzer
from video_editor.color_grader import ColorGrader
from video_editor.pacing_controller import PacingController


class VideoEditor:
    """A class for automated video editing using ffmpeg."""

    def __init__(self, config: Config):
        """Initialize the VideoEditor with configuration.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = self._setup_logger()
        self.clip_analyzer = ClipAnalyzer()
        self.pacing_controller = PacingController()
        self.color_grader = ColorGrader()

        # Set up input/output paths from nested config
        self.input_dir = Path(self.config.input_output.input_videos_dir)
        self.output_dir = Path(self.config.input_output.output_videos_dir)

        # Create necessary directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        width = int(video_info['width'])
        height = int(video_info['height'])
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
        min_length = self.config.video_duration_limits.min_length
        max_length = self.config.video_duration_limits.max_length

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

    def _prepare_vertical_video(self, input_video: str, video_info: Dict) -> Tuple[ffmpeg.Stream, ffmpeg.Stream]:
        """Handle vertical video according to config settings."""
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
            target_width = self.config.output_resolution.width
            target_height = self.config.output_resolution.height

            if input_width == 0 or input_height == 0:
                raise ValueError("Invalid input dimensions")

            self.logger.debug(f"Input dimensions: {input_width}x{input_height}")
            self.logger.debug(f"Target dimensions: {target_width}x{target_height}")

            # Scale video maintaining aspect ratio
            scale_factor = min(target_width / input_width, target_height / input_height)
            new_width = int(input_width * scale_factor)
            new_height = int(input_height * scale_factor)

            # Ensure dimensions are even numbers
            new_width = new_width if new_width % 2 == 0 else new_width - 1
            new_height = new_height if new_height % 2 == 0 else new_height - 1

            if new_width < 2 or new_height < 2:
                raise ValueError(f"Invalid scaled dimensions: {new_width}x{new_height}")

            self.logger.debug(f"Scaling to: {new_width}x{new_height}")
            processed_video = (
                video_stream
                .filter('scale', str(new_width), str(new_height))
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
            if not self.config.audio.normalize:
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
            duration = self.config.transitions.duration

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

                    # Prepare video stream
                    video_stream, audio_stream = self._prepare_vertical_video(video_path, video_info)

                    # Apply audio normalization if available
                    if audio_stream and self.config.audio.normalize:
                        audio_stream = self._normalize_audio(audio_stream)

                    # Apply color grading if enabled
                    if self.config.effects.color_grading.enabled:
                        video_stream = self.color_grader.apply_grading(video_stream)

                    # Output processed video
                    output_path = self.temp_dir / f"processed_{i}.mp4"

                    # Combine streams and output
                    streams = [video_stream]
                    if audio_stream:
                        streams.append(audio_stream)

                    # Configure output options
                    output_options = {
                        'vcodec': 'libx264',
                        'preset': 'medium',
                        'pix_fmt': 'yuv420p',
                        'movflags': '+faststart'
                    }

                    if audio_stream:
                        output_options['acodec'] = 'aac'
                        output_options['ac'] = '2'  # Stereo audio
                    else:
                        output_options['an'] = None  # No audio flag

                    # Create output stream with options
                    stream = ffmpeg.output(
                        *streams,
                        str(output_path),
                        **output_options
                    ).overwrite_output()

                    # Run FFmpeg with error capture
                    try:
                        stdout, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                        processed_videos.append(str(output_path))
                        self.logger.info(f"Successfully processed video {i+1}/{len(filtered_videos)}")
                    except ffmpeg.Error as e:
                        self.logger.error(f"FFmpeg error processing {video_path}:")
                        self.logger.error(f"stdout: {e.stdout.decode() if e.stdout else ''}")
                        self.logger.error(f"stderr: {e.stderr.decode() if e.stderr else ''}")
                        continue

                except Exception as e:
                    self.logger.error(f"Error processing video {video_path}: {str(e)}")
                    continue

            if not processed_videos:
                raise ValueError("No videos were successfully processed")

            # Create final compilation with transitions
            final_output = self.output_dir / "final_compilation.mp4"

            try:
                if len(processed_videos) == 1:
                    shutil.copy2(processed_videos[0], final_output)
                else:
                    # Create concat file for FFmpeg
                    concat_file = self.temp_dir / 'concat.txt'
                    with open(concat_file, 'w') as f:
                        for video in processed_videos:
                            f.write(f"file '{Path(video).absolute()}'\n")

                    # Use concat demuxer for efficient concatenation
                    try:
                        stream = ffmpeg.input(str(concat_file), format='concat', safe=0)
                        stream = ffmpeg.output(
                            stream,
                            str(final_output),
                            acodec='copy',
                            vcodec='copy',
                            movflags='+faststart'
                        ).overwrite_output()

                        stdout, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                        self.logger.debug("Concatenation stdout: " + stdout.decode())
                        self.logger.debug("Concatenation stderr: " + stderr.decode())

                    except ffmpeg.Error as e:
                        self.logger.error("FFmpeg concatenation error:")
                        self.logger.error(f"stdout: {e.stdout.decode() if e.stdout else ''}")
                        self.logger.error(f"stderr: {e.stderr.decode() if e.stderr else ''}")
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
