"""ffmpeg-longrun — run FFmpeg jobs that take hours, and fail loudly when they break.

A thin layer over ``subprocess`` for long encodes: stall detection instead of
timeout guessing, progress telemetry, partial-output cleanup, ffprobe
integrity validation, NVENC probing with an x265 fallback, and an
attenuate-only loudness path.

Nothing here is a pipeline. Every function takes an ffmpeg command (or the
pieces of one) and hands back a result — you keep control of the command.
"""

from .config import (
    DEFAULT_LOUDNESS_TARGETS,
    DEFAULT_VIDEO_PROFILE,
    VERTICAL_SOCIAL_VIDEO_PROFILE,
    LoudnessTargets,
    VideoProfile,
)
from .encoders import (
    NVENCUnavailableError,
    build_cpu_args,
    build_encode_args,
    build_nvenc_args,
    nvenc_available,
    require_nvenc,
)
from .loudness import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)
from .probe import FFprobeClient, FFprobeError
from .runner import (
    FFmpegResult,
    check_disk_space,
    run_ffmpeg_encode,
    run_ffmpeg_quick,
)
from .validate import VideoValidation, validate_video

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # runner
    "FFmpegResult",
    "run_ffmpeg_encode",
    "run_ffmpeg_quick",
    "check_disk_space",
    # validation
    "VideoValidation",
    "validate_video",
    # probing
    "FFprobeClient",
    "FFprobeError",
    # encoders
    "NVENCUnavailableError",
    "nvenc_available",
    "require_nvenc",
    "build_encode_args",
    "build_nvenc_args",
    "build_cpu_args",
    # loudness
    "build_loudnorm_measure_cmd",
    "parse_loudnorm_stats",
    "build_attenuate_filter",
    # config
    "VideoProfile",
    "LoudnessTargets",
    "DEFAULT_VIDEO_PROFILE",
    "VERTICAL_SOCIAL_VIDEO_PROFILE",
    "DEFAULT_LOUDNESS_TARGETS",
]
