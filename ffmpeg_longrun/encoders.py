"""NVENC capability probing and encoder argument builders.

Two things live here:

:func:`nvenc_available`
    A real probe. It asks ffmpeg to encode one tiny synthetic frame with the
    hardware encoder and reports whether that worked. Checking for an NVIDIA
    GPU, or grepping ``ffmpeg -encoders``, both lie: the encoder can be
    compiled in and listed while the driver refuses to open a session (no GPU
    on this host, driver/runtime version mismatch, every NVENC session already
    in use, a container without the device mapped through).

:func:`build_encode_args`
    Produces the ``-c:v ...`` half of an ffmpeg command from a
    :class:`~ffmpeg_longrun.config.VideoProfile`, choosing the NVENC or the
    CPU (x265) argument set. Same quality target either way; the CPU path is
    slower and lands a few percent behind on quality at the same nominal
    setting.

Hardware requirement
--------------------
Everything NVENC needs an NVIDIA GPU with a supported driver and an ffmpeg
built with ``--enable-nvenc``. There is no AMD or Intel path here. Without
one, :func:`nvenc_available` returns ``False`` and :func:`build_encode_args`
emits the x265 arguments — the encode still runs, it just runs on the CPU and
takes considerably longer. Callers that would rather stop than silently spend
hours on a CPU encode should call :func:`require_nvenc` at startup.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Literal

from .config import DEFAULT_VIDEO_PROFILE, VideoProfile

logger = logging.getLogger(__name__)

__all__ = [
    "NVENCUnavailableError",
    "nvenc_available",
    "require_nvenc",
    "build_encode_args",
    "build_nvenc_args",
    "build_cpu_args",
]

AudioMode = Literal["copy", "encode", "none"]


class NVENCUnavailableError(RuntimeError):
    """The NVENC probe failed and the caller asked to treat that as fatal."""


def nvenc_available(
    *,
    codec: str = "hevc_nvenc",
    ffmpeg_bin: str = "ffmpeg",
    timeout: int = 10,
) -> bool:
    """Return ``True`` if *codec* can actually encode on this machine.

    Encodes a single frame of ``testsrc`` to null. The test source is 256x256
    rather than something smaller because NVENC rejects frames below a
    driver-dependent minimum (around 144x144) and would fail the probe for a
    reason that has nothing to do with availability.

    Any failure — non-zero exit, timeout, ffmpeg not on PATH — returns
    ``False``. Use :func:`require_nvenc` when a missing GPU should stop the
    program instead.
    """
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=256x256:rate=1",
                "-c:v",
                codec,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.info("%s probe failed (%s) — CPU encoding will be used", codec, type(e).__name__)
        return False

    if result.returncode != 0:
        logger.info("%s probe returned %d — CPU encoding will be used", codec, result.returncode)
        return False
    return True


def require_nvenc(
    *,
    codec: str = "hevc_nvenc",
    ffmpeg_bin: str = "ffmpeg",
    timeout: int = 10,
) -> None:
    """Raise :class:`NVENCUnavailableError` unless *codec* is usable.

    For jobs where a silent fall back to CPU is worse than not starting — a
    CPU encode of a multi-hour 4K source can take most of a day, and the only
    symptom is that the job "is still running".
    """
    if not nvenc_available(codec=codec, ffmpeg_bin=ffmpeg_bin, timeout=timeout):
        raise NVENCUnavailableError(
            f"{codec} is not usable on this machine. It needs an NVIDIA GPU with a "
            f"supported driver and an ffmpeg built with --enable-nvenc. Verify with: "
            f"{ffmpeg_bin} -f lavfi -i testsrc=duration=1:size=256x256:rate=1 "
            f"-c:v {codec} -f null -"
        )


def build_nvenc_args(profile: VideoProfile = DEFAULT_VIDEO_PROFILE) -> list[str]:
    """Return the NVENC video arguments for *profile*.

    Constant-quality VBR with lookahead, spatial and temporal adaptive
    quantisation, B-frames used as references, and NVENC's own two-pass mode.
    """
    args = [
        "-r",
        profile.output_fps,
        "-c:v",
        profile.nvenc_codec,
        "-preset",
        profile.nvenc_preset,
        "-tune",
        profile.nvenc_tune,
        "-rc",
        "vbr",
        "-cq",
        str(profile.nvenc_cq),
        "-profile:v",
        profile.encoder_profile,
        "-pix_fmt",
        profile.nvenc_pix_fmt,
        "-rc-lookahead",
        str(profile.nvenc_lookahead),
        "-spatial-aq",
        "1" if profile.nvenc_spatial_aq else "0",
        "-temporal-aq",
        "1" if profile.nvenc_temporal_aq else "0",
        "-b_ref_mode",
        profile.nvenc_b_ref_mode,
        "-bf",
        str(profile.nvenc_b_frames),
        "-multipass",
        profile.nvenc_multipass,
        "-g",
        str(profile.gop_size),
    ]
    return args


def build_cpu_args(profile: VideoProfile = DEFAULT_VIDEO_PROFILE) -> list[str]:
    """Return the x265 (CPU) video arguments for *profile*.

    The ``-x265-params`` block pins the keyframe interval on both ends and
    disables open GOP, so the output segments cleanly for streaming; the
    thread settings suit a many-core machine.
    """
    gop = profile.gop_size
    return [
        "-r",
        profile.output_fps,
        "-c:v",
        profile.cpu_codec,
        "-preset",
        profile.cpu_preset,
        "-crf",
        str(profile.cpu_crf),
        "-profile:v",
        profile.encoder_profile,
        "-pix_fmt",
        profile.cpu_pix_fmt,
        "-g",
        str(gop),
        "-x265-params",
        f"keyint={gop}:min-keyint={gop}:open-gop=0:pools=16:frame-threads=4:wpp=1",
    ]


def build_encode_args(
    height: int,
    *,
    has_nvenc: bool,
    profile: VideoProfile = DEFAULT_VIDEO_PROFILE,
    audio: AudioMode = "copy",
) -> list[str]:
    """Build the encoder half of an ffmpeg command.

    Parameters
    ----------
    height:
        Output height in pixels. Selects the VBR bitrate floor.
    has_nvenc:
        Result of :func:`nvenc_available`, or the caller's own decision.
        ``False`` selects the x265 arguments.
    profile:
        Quality settings. See :class:`~ffmpeg_longrun.config.VideoProfile`.
    audio:
        ``"copy"`` stream-copies the source audio (the default — cheapest, and
        correct when loudness was already handled upstream), ``"encode"``
        re-encodes it with the profile's audio codec and bitrate, ``"none"``
        emits ``-an`` and ships no audio stream at all.

    Returns
    -------
    list[str]
        Arguments to append after the input options and before the output
        path. The input arguments (``-i``, seeks, filters) stay the caller's
        business.

    Examples
    --------
    >>> args = build_encode_args(2160, has_nvenc=False)
    >>> args[:4]
    ['-r', '30000/1001', '-c:v', 'libx265']
    >>> "-an" in build_encode_args(1920, has_nvenc=False, audio="none")
    True
    """
    if not has_nvenc:
        logger.warning(
            "%s unavailable; falling back to %s (CPU encode — substantially slower)",
            profile.nvenc_codec,
            profile.cpu_codec,
        )
    args = build_nvenc_args(profile) if has_nvenc else build_cpu_args(profile)

    # Colour metadata tags. These describe the signal; they do not convert it.
    args.extend(
        [
            "-color_range",
            profile.color_range,
            "-colorspace",
            profile.colorspace,
            "-color_primaries",
            profile.color_primaries,
            "-color_trc",
            profile.color_trc,
        ]
    )

    # VBR floor hints. Constant-quality encoding adapts to scene complexity on
    # its own; the floor only stops the encoder from under-spending on simple
    # content, and the maxrate headroom stops it pegging at the ceiling on
    # complex content.
    target, maxrate, bufsize = profile.rate_control_kbps(height)
    args.extend(
        [
            "-b:v",
            f"{target}k",
            "-maxrate",
            f"{maxrate}k",
            "-bufsize",
            f"{bufsize}k",
        ]
    )

    if audio == "encode":
        args.extend(["-c:a", profile.audio_codec, "-b:a", profile.audio_bitrate])
    elif audio == "none":
        args.append("-an")
    elif audio == "copy":
        args.extend(["-c:a", "copy"])
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"audio must be 'copy', 'encode', or 'none' — got {audio!r}")

    if profile.faststart:
        args.extend(["-movflags", "+faststart"])

    return args
