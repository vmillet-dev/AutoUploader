import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class PacingMetrics:
    """Data class to store pacing analysis metrics."""
    path: str
    original_duration: float
    target_duration: float
    speed_factor: float
    transition_duration: float

class PacingController:
    """Class for controlling video pacing and timing."""

    def __init__(self):
        """Initialize the PacingController."""
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('PacingController')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_pacing(self, video_path: str, target_duration: Optional[float] = None) -> PacingMetrics:
        """Analyze video and determine optimal pacing adjustments."""
        # Get video information
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])

        # Calculate speed factor based on content type
        speed_factor = self._calculate_speed_factor(video_path)

        # Adjust transition duration based on content
        transition_duration = self._calculate_transition_duration(video_path)

        # Calculate target duration if not specified
        if target_duration is None:
            target_duration = duration * speed_factor

        return PacingMetrics(
            path=video_path,
            original_duration=duration,
            target_duration=target_duration,
            speed_factor=speed_factor,
            transition_duration=transition_duration
        )

    def _calculate_speed_factor(self, video_path: str) -> float:
        """Calculate optimal speed factor based on content analysis."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            return 1.0

        motion_scores = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)

            prev_frame = gray

        cap.release()

        if not motion_scores:
            return 1.0

        # Calculate speed factor based on motion
        avg_motion = np.mean(motion_scores)
        if avg_motion < 5:  # Low motion
            return 1.5  # Speed up
        elif avg_motion > 20:  # High motion
            return 0.8  # Slow down
        else:
            return 1.0  # Keep original speed

    def _calculate_transition_duration(self, video_path: str) -> float:
        """Calculate optimal transition duration based on content."""
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])

        # Base transition duration on video length
        if duration < 15:
            return 0.5
        elif duration < 30:
            return 0.8
        else:
            return 1.0

    def adjust_clip_speed(self, video_path: str, metrics: PacingMetrics) -> str:
        """Adjust video speed according to pacing metrics."""
        # Check if video is already paced
        if Path(video_path).stem.startswith('paced_'):
            return video_path

        # Create temp directory if it doesn't exist
        temp_dir = Path(video_path).parent / 'temp_paced'
        temp_dir.mkdir(exist_ok=True)

        output_path = str(temp_dir / f"paced_{Path(video_path).stem}.mp4")

        try:
            # Check if video has audio stream
            probe = ffmpeg.probe(video_path)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

            # Apply speed adjustment using FFmpeg
            stream = ffmpeg.input(video_path)

            if metrics.speed_factor != 1.0:
                # Apply setpts filter to video for speed adjustment
                video = stream.video.filter('setpts', f'{1/metrics.speed_factor}*PTS')

                if has_audio:
                    # Apply tempo filter to audio to maintain pitch
                    audio = stream.audio.filter('atempo', metrics.speed_factor)
                    # Combine streams with audio
                    stream = ffmpeg.output(
                        video, audio,
                        output_path,
                        acodec='aac',
                        vcodec='libx264',
                        preset='medium'
                    )
                else:
                    # Output video only
                    stream = ffmpeg.output(
                        video,
                        output_path,
                        vcodec='libx264',
                        preset='medium'
                    )
            else:
                # If no speed adjustment needed, just copy
                stream = ffmpeg.output(
                    stream,
                    output_path,
                    acodec='copy' if has_audio else None,
                    vcodec='copy'
                )

            self.logger.info(f"Adjusting speed of {video_path} by factor {metrics.speed_factor}")
            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            return output_path

        except ffmpeg.Error as e:
            self.logger.error(f"Error adjusting speed for {video_path}: {str(e)}")
            if Path(output_path).exists():
                Path(output_path).unlink()
            raise

    def process_clips(self, video_paths: List[str], target_duration: Optional[float] = None) -> List[Tuple[str, PacingMetrics]]:
        """Process multiple clips and adjust their pacing."""
        processed_clips = []

        for video_path in video_paths:
            # Analyze clip
            metrics = self.analyze_pacing(video_path, target_duration)

            # Adjust speed if needed
            if metrics.speed_factor != 1.0:
                processed_path = self.adjust_clip_speed(video_path, metrics)
            else:
                processed_path = video_path

            processed_clips.append((processed_path, metrics))

        return processed_clips
