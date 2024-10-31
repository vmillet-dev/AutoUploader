import cv2
import numpy as np
import ffmpeg
from typing import Tuple
import logging
from dataclasses import dataclass

@dataclass
class ColorMetrics:
    """Data class to store color analysis metrics."""
    path: str
    brightness: float
    contrast: float
    saturation: float
    white_balance: Tuple[float, float, float]

class ColorGrader:
    """Class for automated color grading and visual enhancement."""

    def __init__(self):
        """Initialize the ColorGrader."""
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ColorGrader')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_colors(self, video_path: str) -> ColorMetrics:
        """Analyze video colors and determine optimal adjustments."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            raise ValueError(f"Could not open video: {video_path}")

        brightness_values = []
        contrast_values = []
        saturation_values = []
        wb_values = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Calculate metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            saturation = np.mean(hsv[:, :, 1])

            # Calculate white balance
            b, g, r = cv2.split(frame)
            wb = (np.mean(b), np.mean(g), np.mean(r))

            brightness_values.append(brightness)
            contrast_values.append(contrast)
            saturation_values.append(saturation)
            wb_values.append(wb)

        cap.release()

        # Calculate average metrics
        avg_brightness = np.mean(brightness_values)
        avg_contrast = np.mean(contrast_values)
        avg_saturation = np.mean(saturation_values)
        avg_wb = tuple(np.mean(wb_values, axis=0))

        return ColorMetrics(
            path=video_path,
            brightness=avg_brightness,
            contrast=avg_contrast,
            saturation=avg_saturation,
            white_balance=avg_wb
        )

    def apply_grading(self, video_stream: ffmpeg.Stream) -> ffmpeg.Stream:
        """Apply color grading and visual enhancements to a video stream."""
        try:
            # Use simple preset values for basic enhancement
            brightness_adj = 0.0  # Neutral brightness
            contrast_adj = 1.1    # Slightly increased contrast
            saturation_adj = 1.1  # Slightly increased saturation

            try:
                # Apply simple color adjustments with safe parameters
                video = video_stream.filter('eq',
                    brightness=str(brightness_adj),
                    contrast=str(contrast_adj),
                    saturation=str(saturation_adj)
                )

                # Log filter parameters
                self.logger.debug("Applied color adjustments")

                # Add minimal sharpening for enhanced detail
                video = video.filter('unsharp',
                    luma_msize_x=3,
                    luma_msize_y=3,
                    luma_amount=0.2  # Reduced for faster processing
                )

                return video

            except ValueError as ve:
                self.logger.warning(f"Error applying color adjustments: {str(ve)}")
                # Return original stream if enhancement fails
                return video_stream

        except Exception as e:
            self.logger.error(f"Error in color grading: {str(e)}")
            # Return original stream if enhancement fails
            return video_stream

    def _calculate_brightness_adjustment(self, brightness: float) -> float:
        """Calculate optimal brightness adjustment."""
        target_brightness = 127.5  # Middle of 0-255 range
        current = max(1.0, brightness)  # Prevent division by zero
        if current < target_brightness:
            return min(1.2, target_brightness / current)
        elif current > target_brightness:
            return max(0.8, target_brightness / current)
        return 1.0

    def _calculate_contrast_adjustment(self, contrast: float) -> float:
        """Calculate optimal contrast adjustment."""
        target_contrast = 50  # Typical standard deviation for good contrast
        current = max(0.1, contrast)  # Prevent division by zero with small threshold
        if current < target_contrast:
            return min(1.3, target_contrast / current)
        elif current > target_contrast:
            return max(0.7, target_contrast / current)
        return 1.0

    def _calculate_saturation_adjustment(self, saturation: float) -> float:
        """Calculate optimal saturation adjustment."""
        target_saturation = 128  # Middle of HSV saturation range
        current = max(0.1, saturation)  # Prevent division by zero with small threshold
        if current < target_saturation:
            return min(1.3, target_saturation / current)
        elif current > target_saturation:
            return max(0.7, target_saturation / current)
        return 1.0

    def _calculate_wb_adjustment(self, wb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate white balance adjustments for RGB channels."""
        # Calculate average intensity across channels
        avg_intensity = sum(wb) / 3

        # Calculate adjustment factors with zero protection
        r_adj = avg_intensity / max(0.1, wb[2])  # Prevent division by zero
        g_adj = avg_intensity / max(0.1, wb[1])
        b_adj = avg_intensity / max(0.1, wb[0])

        # Normalize to -1 to 1 range for FFmpeg colorbalance filter
        r_adj = max(-1.0, min(1.0, (r_adj - 1.0)))
        g_adj = max(-1.0, min(1.0, (g_adj - 1.0)))
        b_adj = max(-1.0, min(1.0, (b_adj - 1.0)))

        return (r_adj, g_adj, b_adj)
