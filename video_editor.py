import yaml
import ffmpeg
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from clip_analyzer import ClipAnalyzer
from pacing_controller import PacingController
from color_grader import ColorGrader
from thumbnail_generator import ThumbnailGenerator

class VideoEditor:
    """A class for automated video editing using ffmpeg."""

    def __init__(self, config: Union[str, Dict]):
        """Initialize the VideoEditor with configuration.

        Args:
            config: Path to YAML configuration file or configuration dictionary
        """
        self.config = self._load_config(config)
        self.logger = self._setup_logger()
        self.clip_analyzer = ClipAnalyzer()
        self.pacing_controller = PacingController()
        self.color_grader = ColorGrader()
        self.thumbnail_generator = ThumbnailGenerator()

    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """Load configuration from YAML file or dictionary."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        elif not isinstance(config, dict):
            raise TypeError("config must be either a file path string or a dictionary")

        # Set default pacing configuration if not present
        if 'pacing' not in config:
            config['pacing'] = {
                'enabled': True,
                'auto_adjust': True,
                'speed_limits': {'min_speed': 0.8, 'max_speed': 1.5},
                'motion_thresholds': {'low_motion': 5.0, 'high_motion': 20.0},
                'transition_duration': {'short_clip': 0.5, 'medium_clip': 0.8, 'long_clip': 1.0}
            }
        # Set default color grading configuration if not present
        if 'color_grading' not in config:
            config['color_grading'] = {
                'enabled': True,
                'auto_adjust': True,
                'target_values': {'brightness': 127.5, 'contrast': 50.0, 'saturation': 128.0},
                'adjustment_limits': {
                    'brightness': [0.8, 1.2],
                    'contrast': [0.7, 1.3],
                    'saturation': [0.7, 1.3],
                    'white_balance': [0.7, 1.3]
                }
            }
        # Set default thumbnail configuration if not present
        if 'thumbnail_generation' not in config:
            config['thumbnail_generation'] = {
                'enabled': True,
                'count': 3,
                'enhancement': {
                    'enabled': True,
                    'contrast': 1.2,
                    'brightness': 1.1,
                    'saturation': 1.2
                }
            }
        return config

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('VideoEditor')
        logger.setLevel(logging.INFO)
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
                # Ensure streams information exists
                if 'streams' not in probe:
                    return {'codec_type': 'video', 'width': 1920, 'height': 1080, 'duration': '0'}
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                # Add streams information to video info
                video_info['streams'] = probe['streams']
                return video_info
            except ffmpeg.Error as e:
                self.logger.warning(f"FFprobe error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to probe video after {max_retries} attempts: {video_path}")
                    return {
                        'codec_type': 'video',
                        'width': 1920,
                        'height': 1080,
                        'duration': '0',
                        'streams': [{'codec_type': 'video', 'width': 1920, 'height': 1080}]
                    }

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

    def _filter_videos_by_duration(self, video_paths: List[str]) -> List[str]:
        """Filter videos based on duration limits from config."""
        try:
            duration_limits = self.config.get('video_duration_limits', {})
            min_length = duration_limits.get('min_length', 5)  # Default to 5 seconds
            max_length = duration_limits.get('max_length', 60)  # Default to 60 seconds
            self.logger.info(f"Filtering videos with duration limits: {min_length}s - {max_length}s")
        except Exception as e:
            self.logger.error(f"Error reading duration limits from config: {str(e)}")
            return []

        # Get durations and scores for all valid videos
        video_info = []
        for video_path in video_paths:
            try:
                duration = self._get_video_duration(video_path)
                if min_length <= duration <= max_length:  # Only check individual duration limits
                    # Get engagement score from clip analyzer
                    if hasattr(self, 'clip_analyzer'):
                        metrics = self.clip_analyzer.analyze_clip(video_path)
                        score = metrics.engagement_score
                    else:
                        score = 1.0
                    video_info.append((video_path, duration, score))
                    self.logger.debug(f"Video {video_path} analyzed: duration={duration:.1f}s, score={score:.3f}")
                else:
                    self.logger.info(f"Skipping {video_path}: Duration {duration:.1f}s outside limits ({min_length}s-{max_length}s)")
            except Exception as e:
                self.logger.error(f"Error processing {video_path}: {str(e)}")
                continue

        if not video_info:
            self.logger.warning("No valid videos found after duration filtering")
            return []

        # Sort by engagement score
        video_info.sort(key=lambda x: float(x[2]), reverse=True)
        self.logger.info(f"Found {len(video_info)} valid videos, sorted by engagement score")

        # Include all videos that meet duration criteria, sorted by engagement
        filtered_videos = []
        total_duration = 0

        for video_path, duration, score in video_info:
            filtered_videos.append(video_path)
            total_duration += duration
            self.logger.info(f"Selected {video_path} (duration: {duration:.1f}s, score: {score:.3f})")

        self.logger.info(f"Final selection: {len(filtered_videos)} videos, total duration: {total_duration:.1f}s")
        return filtered_videos

    def _prepare_vertical_video(self, input_video: str, video_info: Dict) -> Tuple[ffmpeg.Stream, ffmpeg.Stream]:
        """Handle vertical video according to config settings.

        Two modes:
        1. 16:9 output with background filling (maintain_vertical=False)
        2. Vertical output without filling (maintain_vertical=True)
        """
        bg_config = self.config['vertical_video']
        input_stream = ffmpeg.input(input_video)

        # Split video and audio streams
        video_stream = input_stream.video
        audio_stream = input_stream.audio

        # Calculate dimensions
        input_width = int(video_info['width'])
        input_height = int(video_info['height'])
        target_width = self.config['output_resolution']['width']
        target_height = self.config['output_resolution']['height']

        if bg_config.get('maintain_vertical', False):
            # Maintain vertical aspect ratio without filling
            # Scale to target width while maintaining aspect ratio
            scale_factor = target_width / input_width
            new_height = int((input_height * scale_factor) // 2 * 2)

            # Scale video
            processed_video = (
                video_stream
                .filter('scale', str(target_width), str(new_height))
                .filter('format', 'yuv420p')
            )

            self.logger.debug(
                f"Prepared vertical video (maintain vertical): "
                f"original={input_width}x{input_height}, "
                f"scaled={target_width}x{new_height}"
            )
        else:
            # Force 16:9 with background filling
            # Calculate scaling to fit height while maintaining aspect ratio
            scale_factor = target_height / input_height
            new_width = int((input_width * scale_factor) // 2 * 2)
            new_height = target_height

            # Create temporary directory for processing
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()

            try:
                # Create base 16:9 frame
                base = ffmpeg.input('color=c=black:s={}x{}:d=1'.format(target_width, target_height),
                                  f='lavfi')

                if bg_config['background_type'] == 'blur':
                    # Create simple blurred background
                    bg = (
                        video_stream
                        .filter('scale', str(target_width), '-1')
                        .filter('crop', str(target_width), str(target_height))
                        .filter('boxblur', str(bg_config['blur_amount']))
                    )
                elif bg_config['background_type'] == 'image':
                    if not bg_config.get('background_path'):
                        raise ValueError("Background path not specified in config")
                    # Load and scale image background
                    bg = (
                        ffmpeg.input(bg_config['background_path'])
                        .filter('scale', str(target_width), str(target_height))
                    )
                elif bg_config['background_type'] == 'video':
                    if not bg_config.get('background_path'):
                        raise ValueError("Background path not specified in config")
                    # Use background video directly
                    bg = (
                        ffmpeg.input(bg_config['background_path'])
                        .filter('scale', str(target_width), str(target_height))
                    )
                else:
                    bg = base

                # Scale vertical video
                scaled_video = (
                    video_stream
                    .filter('scale', str(new_width), str(new_height))
                )

                # Overlay scaled video on background at center position
                processed_video = (
                    ffmpeg.overlay(bg, scaled_video, x=f"({target_width}-{new_width})/2", y=0)
                    .filter('format', 'yuv420p')
                )

                self.logger.debug(
                    f"Prepared vertical video (16:9 with background): "
                    f"original={input_width}x{input_height}, "
                    f"scaled={new_width}x{new_height}, "
                    f"final={target_width}x{target_height}"
                )

            finally:
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        return processed_video, audio_stream

    def _normalize_audio(self, audio_stream: Optional[ffmpeg.Stream]) -> Optional[ffmpeg.Stream]:
        """Apply audio normalization according to config.

        Args:
            audio_stream: The audio stream to normalize, or None if no audio.

        Returns:
            Normalized audio stream or None if input was None.
        """
        if audio_stream is None:
            self.logger.warning("No audio stream to normalize")
            return None

        try:
            audio_config = self.config.get('audio', {})
            if not audio_config.get('normalize', False):
                return audio_stream

            target_loudness = audio_config.get('target_loudness', -14)
            max_loudness = audio_config.get('max_loudness', -1.0)

            # Apply audio normalization with proper parameter names
            normalized = (
                audio_stream
                .filter('loudnorm',
                    i=str(target_loudness),
                    lra='7',
                    tp=str(max_loudness)
                )
                .filter('aformat',
                    sample_rates='44100',
                    sample_fmts='fltp',
                    channel_layouts='stereo'
                )
            )
            return normalized
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return audio_stream  # Return original stream if normalization fails

    def _add_watermark(self, video_stream: ffmpeg.Stream) -> ffmpeg.Stream:
        """Add watermark to video stream."""
        try:
            watermark_config = self.config.get('watermark', {})
            if not isinstance(watermark_config, dict) or not watermark_config.get('enabled', False):
                return video_stream

            # Get watermark parameters with safe defaults
            position = str(watermark_config.get('position', 'bottom_right'))
            opacity = float(watermark_config.get('opacity', 0.8))
            scale = float(watermark_config.get('scale', 0.2))

            # Calculate padding (10% of smallest video dimension)
            padding = f"min(w,h)*0.1"  # Use expression for dynamic padding

            # Define position expressions
            positions = {
                'top_left': f"{padding}:{padding}",
                'top_right': f"W-w-{padding}:{padding}",
                'bottom_left': f"{padding}:H-h-{padding}",
                'bottom_right': f"W-w-{padding}:H-h-{padding}",
                'center': f"(W-w)/2:(H-h)/2"
            }

            # Get position expression with safe fallback
            overlay_position = positions.get(position, positions['bottom_right'])
            x_pos, y_pos = overlay_position.split(':')

            # Verify watermark file exists
            watermark_path = Path('assets/watermark.png')
            if not watermark_path.exists():
                self.logger.error("Watermark file not found: assets/watermark.png")
                return video_stream

            # Add watermark with validated parameters
            return ffmpeg.overlay(
                video_stream,
                ffmpeg.input(str(watermark_path))
                    .filter('scale', f'iw*{scale}', '-1'),  # Scale watermark
                x=x_pos,
                y=y_pos,
                alpha=opacity
            )
        except Exception as e:
            self.logger.error(f"Error adding watermark: {str(e)}")
            return video_stream

    def _add_text_overlay(self, stream: ffmpeg.Stream, start_time: float = 0) -> ffmpeg.Stream:
        """Add text overlays to video stream."""
        if 'text_overlays' not in self.config:
            return stream

        result = stream
        for overlay in self.config['text_overlays']:
            if not overlay['enabled']:
                continue

            # Calculate text position
            x_pos = {
                'left': f"{overlay['margin']}",
                'center': '(w-text_w)/2',
                'right': f"w-text_w-{overlay['margin']}"
            }[overlay['position']['x']]

            y_pos = {
                'top': f"{overlay['margin']}",
                'center': '(h-text_h)/2',
                'bottom': f"h-text_h-{overlay['margin']}"
            }[overlay['position']['y']]

            # Calculate end time from duration
            start_t = start_time + overlay['start_time']
            duration = overlay.get('duration', float('inf'))
            end_t = start_time + overlay['start_time'] + duration if duration > 0 else float('inf')

            # Add fade effect if enabled
            alpha = '1'
            if overlay.get('fade', {}).get('enabled', False):
                fade_dur = overlay['fade']['duration']
                if duration > 0:
                    alpha = f"if(between(t,{start_t},{start_t+fade_dur}),(t-{start_t})/{fade_dur},if(between(t,{end_t-fade_dur},{end_t}),({end_t}-t)/{fade_dur},1))"
                else:
                    alpha = f"if(between(t,{start_t},{start_t+fade_dur}),(t-{start_t})/{fade_dur},1)"

            # Apply text overlay with timing
            result = result.drawtext(
                text=overlay['text'],
                fontfile=f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                fontsize=overlay['size'],
                fontcolor=overlay['color'],
                x=x_pos,
                y=y_pos,
                alpha=alpha,
                enable=f"between(t,{start_t},{end_t if end_t != float('inf') else 'inf'})"
            )

        return result

    def _apply_transition(self, video1_path: str, video2_path: str, transition_config: dict, output_path: str):
        """Apply transition effect between two videos and save to output file."""
        try:
            duration = self.config['transitions']['duration']
            self.logger.debug(f"Applying {transition_config['type']} transition with duration {duration}s")

            # Input streams with consistent encoding
            stream1 = ffmpeg.input(video1_path)
            stream2 = ffmpeg.input(video2_path)

            # Apply transition based on type
            if transition_config['type'] == 'fade':
                # Fade through black
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition='fade', duration=duration, offset=duration)
            elif transition_config['type'] == 'dissolve':
                # Cross-dissolve effect
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition='fade', duration=duration, offset=duration)
            elif transition_config['type'] == 'wipe':
                # Wipe transition
                direction = transition_config['options']['direction']
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition=f'wipe{direction}', duration=duration, offset=duration)
            elif transition_config['type'] == 'slide':
                # Slide transition
                direction = transition_config['options']['direction']
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition=f'slide{direction}', duration=duration, offset=duration)
            elif transition_config['type'] == 'zoom':
                # Zoom transition
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition='smoothup', duration=duration, offset=duration)
            else:
                # Default to fade if unknown transition type
                self.logger.warning(f"Unknown transition type: {transition_config['type']}, using fade")
                stream = ffmpeg.filter([stream1, stream2], 'xfade',
                    transition='fade', duration=duration, offset=duration)

            # Output with consistent encoding parameters
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec='aac',
                vcodec='libx264',
                preset='ultrafast',
                pix_fmt='yuv420p',
                movflags='+faststart',
                video_bitrate='2000k',
                **{'threads': '4'}
            ).overwrite_output()

            # Run FFmpeg with proper error handling
            process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True)
            try:
                stdout, stderr = process.communicate(timeout=60)  # 1 minute timeout for transitions
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise RuntimeError(f"FFmpeg transition failed: {error_msg}")
                self.logger.debug(f"Successfully applied transition to {output_path}")
            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError("Transition processing timeout")

        except Exception as e:
            self.logger.error(f"Error applying transition: {str(e)}")
            raise

    def _get_random_transition(self) -> dict:
        """Get a random transition configuration from available transitions."""
        transitions = self.config['transitions']['available_transitions']
        return random.choice(transitions)

    def process_videos(self) -> Optional[str]:
        """Process videos in two passes: basic processing and transitions.

        Returns:
            Path to the output video file or None if processing fails
        """
        # Create base output directory
        input_dir = Path(self.config['input_directory'])
        output_dir = Path(self.config['output_directory'])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Created base output directory")

        # Create directory structure for different output formats
        self.vertical_dir = output_dir / 'vertical'
        self.landscape_dir = output_dir / '16_9'
        self.vertical_dir.mkdir(parents=True, exist_ok=True)
        self.landscape_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Created output format directories")

        # Initialize thumbnail directories
        self.vertical_thumbnails = self.vertical_dir / 'thumbnails'
        self.landscape_thumbnails = self.landscape_dir / 'thumbnails'
        self.vertical_thumbnails.mkdir(parents=True, exist_ok=True)
        self.landscape_thumbnails.mkdir(parents=True, exist_ok=True)
        self.logger.info("Created thumbnail directories")

        # Create temporary directories for processing
        self.temp_dir = output_dir / 'temp'
        self.first_pass_dir = self.temp_dir / 'first_pass'
        self.second_pass_dir = self.temp_dir / 'second_pass'
        self.progress_dir = self.temp_dir / 'progress'

        # Create all temporary directories at once
        temp_dirs = [
            self.temp_dir,
            self.first_pass_dir,
            self.second_pass_dir,
            self.progress_dir
        ]
        for dir_path in temp_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.info("Created temporary processing directories")

        # Get and filter videos
        videos = list(input_dir.glob('*.mp4'))
        valid_videos = self._filter_videos_by_duration(map(str, videos))

        if not valid_videos:
            self.logger.error("No valid videos found for processing")
            return None

        # Analyze and order clips
        self.logger.info("Analyzing clips for optimal ordering...")
        ordered_videos = self.clip_analyzer.order_clips(valid_videos)
        self.logger.info("Clip analysis complete")
        first_pass_files = []
        second_pass_files = []
        temp_dirs = [self.temp_dir, self.first_pass_dir, self.second_pass_dir, self.progress_dir]

        try:
            # First pass: Basic scaling and format conversion
            self.logger.info("Starting first pass processing...")
            for i, video_path in enumerate(ordered_videos):
                self.logger.info(f"First pass processing video {i+1}/{len(ordered_videos)}: {video_path}")
                first_pass_output = self.first_pass_dir / f'scaled_{i}.mp4'
                progress_file = self.progress_dir / f'progress_first_{i}.txt'

                try:
                    # Get video information
                    video_info = self._get_video_info(video_path)
                    has_audio = any(s['codec_type'] == 'audio' for s in video_info['streams'])

                    # Setup basic streams
                    input_stream = ffmpeg.input(video_path)
                    video_stream = input_stream.video
                    audio_stream = input_stream.audio if has_audio else None

                    # Basic scaling with optimized filters
                    if self._is_vertical(video_info):
                        if not self.config['vertical_video'].get('maintain_vertical', False):
                            # 16:9 output with centered vertical video
                            target_width = self.config['output_resolution']['width']
                            target_height = self.config['output_resolution']['height']
                            video_stream = (
                                video_stream
                                .filter('scale', '-1', str(target_height))
                                .filter('pad', target_width, target_height,
                                       '(ow-iw)/2', '0', 'black')
                            )
                    else:
                        # Scale horizontal video
                        target_width = self.config['output_resolution']['width']
                        target_height = self.config['output_resolution']['height']
                        video_stream = (
                            video_stream
                            .filter('scale', str(target_width), str(target_height))
                        )

                    # Basic output with minimal processing
                    streams = [video_stream]
                    if audio_stream is not None:
                        streams.append(audio_stream)

                    stream = ffmpeg.output(
                        *streams,
                        str(first_pass_output),
                        acodec='aac' if has_audio else None,
                        vcodec='libx264',
                        preset='ultrafast',
                        crf='28',
                        **{'threads': '4'}
                    ).overwrite_output()

                    # Run first pass with progress monitoring
                    cmd = ffmpeg.compile(stream)
                    cmd.extend(['-progress', str(progress_file)])
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate(timeout=60)  # 60 second timeout
                    if process.returncode != 0:
                        raise RuntimeError(f"First pass FFmpeg error: {stderr.decode() if stderr else 'Unknown error'}")
                    first_pass_files.append(first_pass_output)
                    self.logger.info(f"Completed first pass for video {i+1}")

                except Exception as e:
                    self.logger.error(f"First pass error on video {i+1}: {str(e)}")
                    continue

            # Second pass: Apply effects in stages
            self.logger.info("Starting second pass processing...")
            for i, video_path in enumerate(first_pass_files):
                self.logger.info(f"Second pass processing video {i+1}/{len(first_pass_files)}")
                second_pass_output = self.second_pass_dir / f'final_{i}.mp4'
                progress_file = self.progress_dir / f'progress_second_{i}.txt'

                try:
                    input_stream = ffmpeg.input(str(video_path))
                    video_stream = input_stream.video

                    # Check if video has audio stream
                    has_audio = False
                    probe = ffmpeg.probe(str(video_path))
                    audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                    if audio_streams:
                        try:
                            audio_stream = input_stream.audio
                            has_audio = True
                        except (AttributeError, KeyError):
                            audio_stream = None
                            self.logger.warning(f"Audio stream detected but not accessible in video {i+1}")
                    else:
                        audio_stream = None
                        self.logger.info(f"No audio stream found in video {i+1}")

                    # Apply effects in stages
                    if self.config['color_grading']['enabled']:
                        self.logger.info(f"Applying color grading to video {i+1}")
                        video_stream = self.color_grader.apply_grading(video_stream)

                    # Initialize streams list with video
                    streams = [video_stream]

                    # Process audio if available and enabled
                    if has_audio and self.config.get('audio', {}).get('normalize', False):
                        self.logger.info(f"Processing audio for video {i+1}")
                        try:
                            processed_audio = self._normalize_audio(audio_stream)
                            if processed_audio is not None:
                                streams.append(processed_audio)
                                self.logger.info(f"Audio processing successful for video {i+1}")
                        except Exception as e:
                            self.logger.warning(f"Audio processing failed for video {i+1}: {str(e)}")

                    # Apply overlay effects
                    if self.config.get('watermark', {}).get('enabled', False):
                        self.logger.info(f"Adding watermark to video {i+1}")
                        video_stream = self._add_watermark(video_stream)

                    if self.config.get('text_overlay', {}).get('enabled', False):
                        self.logger.info(f"Adding text overlay to video {i+1}")
                        video_stream = self._add_text_overlay(video_stream)

                    # Configure output options based on stream availability
                    output_options = {
                        'vcodec': 'libx264',
                        'preset': 'ultrafast',
                        'crf': '28',
                        'pix_fmt': 'yuv420p',
                        'movflags': '+faststart',
                        'threads': '4'
                    }

                    if has_audio:
                        output_options['acodec'] = 'aac'
                    else:
                        output_options['an'] = None  # No audio flag

                    # Output with effects
                    stream = ffmpeg.output(
                        *streams,
                        str(second_pass_output),
                        **output_options
                    ).overwrite_output()

                    # Run second pass with progress monitoring
                    cmd = ffmpeg.compile(stream)
                    cmd.extend(['-progress', str(progress_file)])
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate(timeout=120)  # 120 second timeout
                    if process.returncode != 0:
                        raise RuntimeError(f"Second pass FFmpeg error: {stderr.decode() if stderr else 'Unknown error'}")
                    second_pass_files.append(second_pass_output)
                    self.logger.info(f"Completed second pass for video {i+1}")

                except Exception as e:
                    self.logger.error(f"Second pass error on video {i+1}: {str(e)}")
                    continue

            # Final concatenation
            if not second_pass_files:
                raise RuntimeError("No videos were successfully processed")

            output_file = output_dir / 'final_compilation.mp4'
            if len(second_pass_files) == 1:
                shutil.copy2(second_pass_files[0], output_file)
            else:
                concat_file = self.temp_dir / 'concat.txt'
                with open(concat_file, 'w') as f:
                    for file in second_pass_files:
                        if file.exists():
                            f.write(f"file '{file.absolute()}'\n")
                        else:
                            self.logger.error(f"Missing processed file: {file}")
                            return None

                try:
                    self.logger.info("Starting final concatenation...")
                    progress_file = self.progress_dir / 'progress_concat.txt'

                    # Use concat demuxer for more efficient concatenation
                    stream = ffmpeg.input(str(concat_file), format='concat', safe=0)
                    stream = ffmpeg.output(
                        stream,
                        str(output_file),
                        acodec='copy',  # Copy audio without re-encoding
                        vcodec='copy',  # Copy video without re-encoding
                        movflags='+faststart',
                        **{'threads': '4'}
                    ).overwrite_output()

                    # Run concatenation with extended timeout and progress monitoring
                    cmd = ffmpeg.compile(stream)
                    cmd.extend(['-progress', str(progress_file)])
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Monitor progress while waiting for completion
                    start_time = time.time()
                    while process.poll() is None:
                        if time.time() - start_time > 180:  # 180 second timeout
                            process.kill()
                            raise TimeoutError("Concatenation timed out after 180 seconds")
                        time.sleep(1)  # Check every second

                    if process.returncode != 0:
                        stderr = process.stderr.read().decode()
                        raise RuntimeError(f"Concatenation failed: {stderr}")

                    self.logger.info("Successfully concatenated videos")

                    # Generate thumbnails if enabled
                    if self.config['thumbnail_generation']['enabled']:
                        self.logger.info("Generating thumbnails...")
                        try:
                            # Select thumbnail directory based on output format
                            thumbnail_dir = self.vertical_thumbnails if self.config.get('maintain_vertical', False) else self.landscape_thumbnails
                            thumbnail_paths = self.thumbnail_generator.generate_thumbnails(
                                str(output_file),
                                str(thumbnail_dir),
                                count=self.config['thumbnail_generation']['count'],
                                enhancement=self.config['thumbnail_generation']['enhancement']
                            )
                            self.logger.info(f"Generated {len(thumbnail_paths)} thumbnails")
                            return str(output_file)
                        except Exception as e:
                            self.logger.error(f"Thumbnail generation failed: {str(e)}")
                            # Continue even if thumbnail generation fails
                            return str(output_file)

                except Exception as e:
                    self.logger.error(f"Final concatenation failed: {str(e)}")
                    return None

        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            return None

        finally:
            # Cleanup temporary files
            temp_dirs = [self.temp_dir]
            for temp_dir in temp_dirs:
                try:
                    if temp_dir.exists():
                        self.logger.info(f"Cleaning up temporary directory: {temp_dir}")
                        shutil.rmtree(temp_dir)
                        self.logger.info(f"Successfully cleaned up {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {temp_dir}: {str(e)}")
