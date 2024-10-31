import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from scipy.signal import find_peaks

@dataclass
class ClipMetrics:
    """Data class to store clip analysis metrics."""
    path: str
    motion_score: float
    audio_score: float
    scene_complexity: float
    engagement_score: float
    duration: float

class ClipAnalyzer:
    """Class for analyzing video clips and determining optimal ordering."""

    def __init__(self):
        """Initialize the ClipAnalyzer."""
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ClipAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_clip(self, video_path: str) -> ClipMetrics:
        """Analyze a video clip and return its metrics."""
        self.logger.info(f"Analyzing clip: {video_path}")

        # Get video duration
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])

        # Calculate motion score
        motion_score = self._calculate_motion_score(video_path)

        # Calculate audio score
        audio_score = self._calculate_audio_score(video_path)

        # Calculate scene complexity
        scene_complexity = self._calculate_scene_complexity(video_path)

        # Calculate overall engagement score
        engagement_score = self._calculate_engagement_score(
            motion_score, audio_score, scene_complexity
        )

        return ClipMetrics(
            path=video_path,
            motion_score=motion_score,
            audio_score=audio_score,
            scene_complexity=scene_complexity,
            engagement_score=engagement_score,
            duration=duration
        )

    def _calculate_motion_score(self, video_path: str) -> float:
        """Calculate motion score based on frame differences."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            return 0.0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Sample every N frames (process 1 frame per second)
        frame_interval = max(1, fps)
        self.logger.debug(f"Processing {video_path} at {frame_interval} frame intervals")

        prev_frame = None
        motion_scores = []
        frames_processed = 0
        max_processing_frames = min(total_frames, fps * 60)  # Max 60 seconds worth of frames

        while frames_processed < max_processing_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every Nth frame
            if frames_processed % frame_interval == 0:
                try:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if prev_frame is not None:
                        # Calculate frame difference
                        diff = cv2.absdiff(gray, prev_frame)
                        motion_score = np.mean(diff)
                        motion_scores.append(motion_score)

                    prev_frame = gray

                except cv2.error as e:
                    self.logger.warning(f"Error processing frame {frames_processed}: {str(e)}")
                    continue

            frames_processed += 1

        cap.release()
        self.logger.debug(f"Processed {len(motion_scores)} samples from {video_path}")
        return np.mean(motion_scores) if motion_scores else 0.0

    def _calculate_audio_score(self, video_path: str) -> float:
        """Calculate audio score based on volume levels and peaks."""
        try:
            # Extract audio data using ffmpeg
            out, _ = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1)
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Convert to numpy array
            audio_data = np.frombuffer(out, np.float32)

            # Calculate RMS volume
            rms = np.sqrt(np.mean(np.square(audio_data)))

            # Find peaks
            peaks, _ = find_peaks(np.abs(audio_data), height=0.5)
            peak_density = len(peaks) / len(audio_data)

            # Combine metrics
            audio_score = (rms + peak_density) / 2
            return float(audio_score)
        except ffmpeg.Error:
            self.logger.warning(f"No audio stream found in {video_path}")
            return 0.0

    def _calculate_scene_complexity(self, video_path: str) -> float:
        """Calculate scene complexity based on edge detection and color variance."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            return 0.0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Sample every N frames (process 1 frame per second)
        frame_interval = max(1, fps)
        self.logger.debug(f"Processing {video_path} at {frame_interval} frame intervals")

        complexity_scores = []
        frames_processed = 0
        max_processing_frames = min(total_frames, fps * 60)  # Max 60 seconds worth of frames

        while frames_processed < max_processing_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every Nth frame
            if frames_processed % frame_interval == 0:
                try:
                    # Edge detection with reduced resolution for speed
                    small_frame = cv2.resize(frame, (480, 270))
                    edges = cv2.Canny(small_frame, 100, 200)
                    edge_density = np.mean(edges > 0)

                    # Optimized color variance calculation
                    color_vars = np.var(small_frame.reshape(-1, 3), axis=0)
                    color_variance = np.mean(color_vars)

                    # Combine metrics
                    frame_complexity = (edge_density + np.log1p(color_variance)) / 2
                    complexity_scores.append(frame_complexity)

                except cv2.error as e:
                    self.logger.warning(f"Error processing frame {frames_processed}: {str(e)}")
                    continue

            frames_processed += 1

        cap.release()
        self.logger.debug(f"Processed {len(complexity_scores)} samples from {video_path}")
        return float(np.mean(complexity_scores)) if complexity_scores else 0.0

    def _calculate_engagement_score(
        self,
        motion_score: float,
        audio_score: float,
        scene_complexity: float
    ) -> float:
        """Calculate overall engagement score."""
        # Weights for different components
        weights = {
            'motion': 0.4,
            'audio': 0.3,
            'complexity': 0.3
        }

        # Normalize scores to 0-1 range
        normalized_motion = np.clip(motion_score / 255.0, 0, 1)
        normalized_audio = np.clip(audio_score, 0, 1)
        normalized_complexity = np.clip(scene_complexity, 0, 1)

        # Calculate weighted sum
        engagement_score = (
            weights['motion'] * normalized_motion +
            weights['audio'] * normalized_audio +
            weights['complexity'] * normalized_complexity
        )

        return float(engagement_score)

    def order_clips(self, clips: List[str]) -> List[str]:
        """Analyze and order clips for maximum engagement."""
        # Filter out enhanced and temporary files
        filtered_clips = []
        for clip in clips:
            clip_path = Path(clip)
            if not any(prefix in clip_path.stem for prefix in ['enhanced_', 'temp_', 'paced_']):
                filtered_clips.append(clip)
                self.logger.debug(f"Including original clip: {clip}")
            else:
                self.logger.debug(f"Filtering out temporary/enhanced clip: {clip}")

        if not filtered_clips:
            raise ValueError("No valid original clips found after filtering")

        # Analyze all clips
        try:
            clip_metrics = [self.analyze_clip(clip) for clip in filtered_clips]
        except Exception as e:
            self.logger.error(f"Error analyzing clips: {str(e)}")
            raise

        # Sort clips by engagement score
        sorted_metrics = sorted(
            clip_metrics,
            key=lambda x: x.engagement_score,
            reverse=True
        )

        # Strategic ordering for maximum viewer retention
        final_order = []
        total_clips = len(sorted_metrics)

        if total_clips <= 1:
            return [clip.path for clip in sorted_metrics]

        # Reserve highest engaging clips for start and end
        final_order.append(sorted_metrics.pop(0))  # Best clip for opening
        end_clip = sorted_metrics.pop(0)  # Second best clip for ending

        # Distribute remaining clips to maintain engagement
        while len(sorted_metrics) > 0:
            if len(sorted_metrics) == 1:
                # Add last remaining clip
                final_order.append(sorted_metrics.pop())
            else:
                # Alternate between medium and lower engagement
                mid_idx = len(sorted_metrics) // 2
                # Add medium engagement clip
                final_order.append(sorted_metrics.pop(mid_idx))
                if sorted_metrics:
                    # Add lower engagement clip
                    final_order.append(sorted_metrics.pop())

        # Add the reserved high-engagement clip at the end
        final_order.append(end_clip)

        return [clip.path for clip in final_order]
