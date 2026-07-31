"""Integrity validation for a finished media file.

``validate_video`` answers one question: *did the encode actually produce a
usable file?* It is meant to run immediately after an encode and before
anything irreversible happens to the result — an upload, a concat, a delete of
the source.

What it catches that "ffmpeg exited 0" does not:

* a truncated MP4 with no moov atom (ffmpeg can exit 0 and still leave one
  behind if it is killed at the wrong moment, or if the disk filled),
* a file with no video stream, or no audio stream when one was expected,
* an output whose duration does not match what was asked for — the usual sign
  that a filter or a seek silently dropped part of the timeline,
* audio and video streams whose durations have drifted apart, which a
  container-level duration check misses entirely.

The function never raises for an unusable file or a missing ffprobe. It always
returns a :class:`VideoValidation` whose ``reason`` explains the verdict, so a
caller can log it, retry, or fail the job as it sees fit.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["VideoValidation", "validate_video"]

#: Fractional A/V stream-duration difference treated as drift.
DEFAULT_AV_DRIFT_TOLERANCE = 0.02

#: Fractional deviation from the expected container duration that is accepted.
DEFAULT_DURATION_TOLERANCE = 0.05


@dataclass
class VideoValidation:
    """Result of an ffprobe-based integrity check.

    Attributes
    ----------
    valid:
        ``True`` only when every requested check passed.
    reason:
        Why the file was rejected, or ``None`` when it was accepted.
    duration:
        Container duration in seconds.
    has_video, has_audio:
        Whether a stream of that type is present.
    video_codec:
        ``codec_name`` of the first video stream.
    file_size:
        Size on disk in bytes.
    audio_duration, video_duration:
        Per-stream durations, when the container reports them. Zero when it
        does not — some containers carry only a top-level duration.
    av_drift:
        Absolute difference between the two per-stream durations, in seconds.
    """

    valid: bool
    reason: str | None = None
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    file_size: int = 0
    audio_duration: float = 0.0
    video_duration: float = 0.0
    av_drift: float = 0.0


def validate_video(
    video_path: Path,
    expected_duration: float | None = None,
    duration_tolerance: float = DEFAULT_DURATION_TOLERANCE,
    require_audio: bool = True,
    av_drift_tolerance: float = DEFAULT_AV_DRIFT_TOLERANCE,
    ffprobe_bin: str = "ffprobe",
    timeout: int = 30,
) -> VideoValidation:
    """Validate a media file with ffprobe.

    Checks, in order — the first failure wins:

    1. the file exists and is non-empty,
    2. ffprobe can parse it (this is what catches a missing moov atom),
    3. it has at least one video stream,
    4. it has an audio stream, when *require_audio*,
    5. per-stream audio and video durations agree within *av_drift_tolerance*,
    6. the container duration is within *duration_tolerance* of
       *expected_duration*, when one is given.

    Parameters
    ----------
    video_path:
        File to check.
    expected_duration:
        Duration the caller asked ffmpeg to produce, in seconds. Skipped when
        ``None``.
    duration_tolerance:
        Allowed fractional deviation from *expected_duration*.
    require_audio:
        Reject a file with no audio stream. Set ``False`` for deliberately
        silent output.
    av_drift_tolerance:
        Allowed fractional difference between audio and video stream
        durations. Skipped when the container omits per-stream durations.
    ffprobe_bin:
        ffprobe executable name or path.
    timeout:
        Seconds to wait for ffprobe. A probe that hangs is itself evidence of
        a damaged file.

    Returns
    -------
    VideoValidation
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return VideoValidation(valid=False, reason="File does not exist")

    file_size = video_path.stat().st_size
    if file_size == 0:
        return VideoValidation(valid=False, reason="File is empty (0 bytes)")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return VideoValidation(
            valid=False,
            reason="ffprobe timed out (possible corrupt file)",
            file_size=file_size,
        )
    except FileNotFoundError:
        # A validator that returns a verdict must not raise because a tool is
        # missing — the caller cannot tell that apart from a bad file.
        return VideoValidation(
            valid=False,
            reason=f"{ffprobe_bin} not found on PATH — install ffmpeg to enable validation",
            file_size=file_size,
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        return VideoValidation(
            valid=False,
            reason=f"ffprobe failed: {stderr[:200]}",
            file_size=file_size,
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        return VideoValidation(
            valid=False,
            reason="ffprobe returned invalid JSON",
            file_size=file_size,
        )

    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    video_codec = ""
    for s in streams:
        if s.get("codec_type") == "video":
            video_codec = s.get("codec_name", "")
            break

    if not has_video:
        return VideoValidation(
            valid=False,
            reason="No video stream found",
            has_audio=has_audio,
            video_codec=video_codec,
            file_size=file_size,
        )

    if require_audio and not has_audio:
        return VideoValidation(
            valid=False,
            reason="No audio stream found (required)",
            has_video=has_video,
            video_codec=video_codec,
            file_size=file_size,
        )

    # Per-stream A/V duration drift. A container-level duration check passes a
    # file whose streams have drifted apart, as long as the overall duration
    # still looks right — the symptom is audio that runs ahead of or behind
    # picture, which nothing downstream detects. Some containers omit
    # per-stream "duration"; there we fall back to the container check below.
    video_stream_dur = 0.0
    audio_stream_dur = 0.0
    for s in streams:
        codec_type = s.get("codec_type")
        try:
            stream_dur = float(s.get("duration", 0) or 0)
        except (TypeError, ValueError):
            stream_dur = 0.0
        if codec_type == "video" and stream_dur > 0:
            video_stream_dur = stream_dur
        elif codec_type == "audio" and stream_dur > 0:
            audio_stream_dur = stream_dur

    drift = 0.0
    if video_stream_dur > 0 and audio_stream_dur > 0:
        drift = abs(video_stream_dur - audio_stream_dur)
        longer = max(video_stream_dur, audio_stream_dur)
        drift_pct = drift / longer if longer > 0 else 0.0
        if drift_pct > av_drift_tolerance:
            return VideoValidation(
                valid=False,
                reason=(
                    f"A/V drift: video={video_stream_dur:.2f}s "
                    f"audio={audio_stream_dur:.2f}s "
                    f"(drift={drift:.2f}s, {drift_pct:.1%} > {av_drift_tolerance:.0%})"
                ),
                duration=float(probe.get("format", {}).get("duration", 0)),
                has_video=has_video,
                has_audio=has_audio,
                video_codec=video_codec,
                file_size=file_size,
                audio_duration=audio_stream_dur,
                video_duration=video_stream_dur,
                av_drift=drift,
            )

    duration = float(probe.get("format", {}).get("duration", 0))

    if expected_duration is not None and expected_duration > 0 and duration > 0:
        deviation = abs(duration - expected_duration) / expected_duration
        if deviation > duration_tolerance:
            return VideoValidation(
                valid=False,
                reason=(
                    f"Duration mismatch: {duration:.1f}s vs expected {expected_duration:.1f}s "
                    f"({deviation:.1%} deviation, tolerance {duration_tolerance:.0%})"
                ),
                duration=duration,
                has_video=has_video,
                has_audio=has_audio,
                video_codec=video_codec,
                file_size=file_size,
                audio_duration=audio_stream_dur,
                video_duration=video_stream_dur,
                av_drift=drift,
            )

    return VideoValidation(
        valid=True,
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        video_codec=video_codec,
        file_size=file_size,
        audio_duration=audio_stream_dur,
        video_duration=video_stream_dur,
        av_drift=drift,
    )
