import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import ffmpeg

@dataclass
class ThumbnailMetrics:
    """Data class to store thumbnail analysis metrics."""
    motion_score: float
    face_count: int
    brightness: float
    contrast: float
    composition_score: float

class ThumbnailGenerator:
    """Class for automated thumbnail generation from videos."""

    def __init__(self):
        """Initialize the ThumbnailGenerator."""
        self.logger = self._setup_logger()
        # Load face detection model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Error loading face cascade classifier from {cascade_path}")

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ThumbnailGenerator')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_frame(self, frame: np.ndarray) -> ThumbnailMetrics:
        """Analyze a frame for thumbnail suitability."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face_count = len(faces)

        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Calculate motion score using frame differences
        motion_score = self._calculate_motion_score(frame)

        # Calculate composition score using rule of thirds
        composition_score = self._calculate_composition_score(frame, faces)

        return ThumbnailMetrics(
            motion_score=motion_score,
            face_count=face_count,
            brightness=brightness,
            contrast=contrast,
            composition_score=composition_score
        )

    def _calculate_motion_score(self, frame: np.ndarray) -> float:
        """Calculate motion score using edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.mean(edges)

    def _calculate_composition_score(self, frame: np.ndarray, faces: np.ndarray) -> float:
        """Calculate composition score using rule of thirds."""
        height, width = frame.shape[:2]
        score = 0.0

        # Define rule of thirds points
        third_h = height // 3
        third_w = width // 3
        points = [
            (third_w, third_h),
            (third_w * 2, third_h),
            (third_w, third_h * 2),
            (third_w * 2, third_h * 2)
        ]

        # Score based on face positions relative to thirds points
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_center = (x + w//2, y + h//2)
                for point in points:
                    distance = np.sqrt((face_center[0] - point[0])**2 +
                                     (face_center[1] - point[1])**2)
                    score += 1.0 / (1.0 + distance/100.0)

        return score

    def generate_thumbnails(self, video_path: str, output_dir: str,
                          count: int = 3, enhancement: Optional[dict] = None) -> List[str]:
        """Generate multiple thumbnails from a video."""
        self.logger.info(f"Generating thumbnails for {video_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / fps if fps > 0 else 0

            if duration <= 0:
                raise ValueError("Invalid video duration")

            # Sample frames at different timestamps
            thumbnails = []
            best_frames = []
            # Sample 20 points, avoiding first/last 5 seconds
            sample_frames = np.linspace(
                5 * fps,
                frame_count - (5 * fps),
                20,
                dtype=int
            )
            total_points = len(sample_frames)

            for idx, frame_pos in enumerate(sample_frames, 1):
                self.logger.info(f"Analyzing frame {idx}/{total_points}")

                # Set frame position
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos):
                    self.logger.warning(f"Failed to set position to frame {frame_pos}")
                    continue

                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame at position {frame_pos}")
                    continue

                try:
                    metrics = self.analyze_frame(frame)
                    score = (metrics.motion_score * 0.2 +
                            metrics.face_count * 30.0 +
                            metrics.composition_score * 0.3 +
                            (metrics.brightness/128.0) * 0.2 +
                            (metrics.contrast/50.0) * 0.3)

                    best_frames.append((score, frame))
                except Exception as e:
                    self.logger.error(f"Error analyzing frame at position {frame_pos}: {str(e)}")
                    continue

            cap.release()

            if not best_frames:
                raise ValueError("No valid frames could be analyzed")

            # Sort frames by score and save top N
            best_frames.sort(key=lambda x: x[0], reverse=True)
            for i, (score, frame) in enumerate(best_frames[:count]):
                output_file = output_path / f"thumbnail_{i+1}.jpg"
                if cv2.imwrite(str(output_file), frame):
                    thumbnails.append(str(output_file))
                    self.logger.info(f"Generated thumbnail {i+1} with score {score:.2f}")
                    # Apply enhancement if configured
                    if enhancement and enhancement.get('enabled', False):
                        try:
                            enhanced_path = self.enhance_thumbnail(str(output_file))
                            thumbnails[-1] = enhanced_path  # Replace with enhanced version
                            self.logger.info(f"Enhanced thumbnail {i+1}")
                        except Exception as e:
                            self.logger.error(f"Failed to enhance thumbnail {i+1}: {str(e)}")
                else:
                    self.logger.error(f"Failed to save thumbnail {i+1}")

            return thumbnails

        except Exception as e:
            self.logger.error(f"Error generating thumbnails: {str(e)}")
            raise

    def enhance_thumbnail(self, image_path: str) -> str:
        """Enhance a thumbnail for better visual appeal."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Calculate current brightness and contrast
            brightness = np.mean(image)
            contrast = np.std(image)

            # Adjust enhancement parameters based on current values
            alpha = 1.2 if contrast < 50 else 1.1  # Contrast adjustment
            beta = 10 if brightness < 100 else 5    # Brightness adjustment

            # Apply enhancements
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

            # Save enhanced thumbnail
            output_path = str(Path(image_path).parent / f"enhanced_{Path(image_path).name}")
            if not cv2.imwrite(output_path, enhanced):
                raise IOError(f"Failed to save enhanced thumbnail: {output_path}")

            self.logger.info(f"Enhanced thumbnail saved: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error enhancing thumbnail: {str(e)}")
            raise
