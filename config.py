import os
from dataclasses import dataclass
from dacite import from_dict

import yaml

@dataclass
class IOSettings:
    input_videos_dir: str
    output_videos_dir: str
    video_urls_by_keyword_file: str

# Video Output Settings
@dataclass
class OutputResolution:
    width: int
    height: int
    maintain_aspect_ratio: bool

# Video Selection Criteria
@dataclass
class VideoSelection:
    amount: int  # Number of videos to download per tag

@dataclass
class VideoDurationLimits:
    min_length: int  # in seconds
    max_length: int  # in seconds

# Vertical Video Handling
@dataclass
class VerticalVideo:
    background_type: str  # Options: blur, image, video
    background_path: str  # Path to background image/video if type is image/video
    blur_amount: int  # Amount of blur if background_type is blur

# Audio Settings
@dataclass
class AudioSettings:
    normalize: bool
    target_loudness: float  # Target LUFS
    max_loudness: float     # Maximum peak loudness in dB
    sample_rate: int        # Audio sample rate in Hz

# Text Overlay Settings
@dataclass
class DefaultPosition:
    x: str  # Options: left, center, right, or pixel value
    y: str  # Options: top, center, bottom, or pixel value

@dataclass
class TextOverlay:
    enabled: bool
    default_font: str
    default_size: int
    default_color: str
    default_position: DefaultPosition

# Transition Settings
@dataclass
class Transitions:
    type: str  # Options: fade, dissolve, wipe
    duration: float  # Duration in seconds

# Video Effects Settings
@dataclass
class ColorGrading:
    enabled: bool
    brightness: float
    contrast: float
    saturation: float

@dataclass
class Speed:
    enabled: bool
    factor: float  # 1.0 is normal speed, <1 slower, >1 faster

@dataclass
class Effects:
    color_grading: ColorGrading
    speed: Speed

# Watermark Settings
@dataclass
class WatermarkPosition:
    x: str  # Options: left, center, right, or pixel value
    y: str  # Options: top, center, bottom, or pixel value

@dataclass
class WatermarkSize:
    width: int
    height: int

@dataclass
class Watermark:
    enabled: bool
    path: str  # Path to watermark image
    position: WatermarkPosition
    opacity: float  # 0.0 to 1.0
    size: WatermarkSize

@dataclass
class Config:
    input_output: IOSettings
    output_resolution: OutputResolution
    video_selection: VideoSelection
    video_duration_limits: VideoDurationLimits
    vertical_video: VerticalVideo
    audio: AudioSettings
    text_overlay: TextOverlay
    transitions: Transitions
    effects: Effects
    watermark: Watermark

    @staticmethod
    def load_config(file_path: str):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config = from_dict(data_class=Config, data=config_data)

        for dir_path in config.input_output.__dict__.values():
            if dir_path.endswith('/') or not os.path.splitext(dir_path)[1]:
                os.makedirs(dir_path, exist_ok=True)

        return config
