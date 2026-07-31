"""Tests for ffmpeg_longrun.validate.

ffprobe is mocked for every structural case, so the suite needs no media
files. Two cases deliberately run the real binary and skip when it is absent.
"""

import json
import shutil
import subprocess
from unittest.mock import Mock, patch

import pytest

from ffmpeg_longrun.validate import validate_video

_HAVE_FFPROBE = shutil.which("ffprobe") is not None


def _probe(returncode=0, payload=None, stderr=""):
    return Mock(
        returncode=returncode,
        stdout=json.dumps(payload) if payload is not None else "",
        stderr=stderr,
    )


class TestValidateVideo:
    def test_nonexistent_file(self, tmp_path):
        result = validate_video(tmp_path / "missing.mp4")
        assert not result.valid
        assert "not exist" in result.reason

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        result = validate_video(f)
        assert not result.valid
        assert "empty" in result.reason

    @pytest.mark.skipif(not _HAVE_FFPROBE, reason="ffprobe not installed")
    def test_truncated_file_no_moov(self, tmp_path):
        """A file that is not valid MP4 fails the real ffprobe parse.

        This is the moov-atom case: the bytes are there, the container header
        is not, and only an actual parse catches it.
        """
        f = tmp_path / "truncated.mp4"
        f.write_bytes(b"\x00" * 1024)

        result = validate_video(f)
        assert not result.valid
        assert result.reason is not None

    def test_missing_ffprobe_returns_verdict_not_exception(self, tmp_path):
        """A missing ffprobe yields an invalid verdict, never a raised error."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"fake")

        with patch("ffmpeg_longrun.validate.subprocess.run", side_effect=FileNotFoundError):
            result = validate_video(f)

        assert not result.valid
        assert "not found on PATH" in result.reason

    def test_ffprobe_timeout_is_reported(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"fake")

        with patch(
            "ffmpeg_longrun.validate.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
        ):
            result = validate_video(f)

        assert not result.valid
        assert "timed out" in result.reason

    def test_ffprobe_nonzero_exit_is_reported(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"fake")

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(returncode=1, stderr="moov atom not found")
            result = validate_video(f)

        assert not result.valid
        assert "moov atom not found" in result.reason

    def test_invalid_json_is_reported(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"fake")

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = Mock(returncode=0, stdout="not json at all", stderr="")
            result = validate_video(f)

        assert not result.valid
        assert "invalid JSON" in result.reason

    def test_valid_video(self, tmp_path):
        f = tmp_path / "good.mp4"
        f.write_bytes(b"fake video")

        payload = {
            "format": {"duration": "120.5"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert result.valid
        assert result.duration == pytest.approx(120.5)
        assert result.has_video
        assert result.has_audio
        assert result.video_codec == "h264"
        assert result.file_size == len(b"fake video")

    def test_duration_mismatch_fails(self, tmp_path):
        f = tmp_path / "short.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "60.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, expected_duration=120.0, duration_tolerance=0.05)

        assert not result.valid
        assert "Duration mismatch" in result.reason

    def test_duration_within_tolerance_passes(self, tmp_path):
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "118.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, expected_duration=120.0, duration_tolerance=0.05)

        assert result.valid

    def test_missing_audio_when_required_fails(self, tmp_path):
        f = tmp_path / "noaudio.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "120.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, require_audio=True)

        assert not result.valid
        assert "audio" in result.reason.lower()

    def test_missing_audio_when_not_required_passes(self, tmp_path):
        f = tmp_path / "noaudio.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "120.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, require_audio=False)

        assert result.valid

    def test_no_video_stream_fails(self, tmp_path):
        f = tmp_path / "audioonly.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "120.0"},
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert not result.valid
        assert "video" in result.reason.lower()


class TestValidateVideoAVDrift:
    """Per-stream A/V duration drift detection.

    A container-level duration check alone passes a file whose audio and video
    streams have drifted apart, as long as the overall duration still matches
    expectations. These cover the per-stream comparison.
    """

    def test_av_drift_within_tolerance_passes(self, tmp_path):
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "30.1"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "30.2"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert result.valid
        assert result.av_drift == pytest.approx(0.2, abs=0.01)
        assert result.video_duration == pytest.approx(30.0)
        assert result.audio_duration == pytest.approx(30.2)

    def test_av_drift_exceeding_threshold_fails(self, tmp_path):
        f = tmp_path / "drift.mp4"
        f.write_bytes(b"fake")

        # video 30s / audio 24.5s → drift 5.5s / 30s = 18.3% — well above 2%
        payload = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "24.5"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert not result.valid
        assert "A/V drift" in result.reason
        assert result.av_drift == pytest.approx(5.5, abs=0.01)

    def test_av_drift_tolerance_is_configurable(self, tmp_path):
        """A caller that expects loose sync can widen the tolerance."""
        f = tmp_path / "drift.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "24.5"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, av_drift_tolerance=0.25)

        assert result.valid

    def test_av_drift_check_skipped_when_no_audio(self, tmp_path):
        f = tmp_path / "videoonly.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "30.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "duration": "30.0"}],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f, require_audio=False)

        assert result.valid
        assert result.av_drift == 0.0

    def test_av_drift_check_skipped_when_stream_duration_missing(self, tmp_path):
        """Container-only duration is common for some streaming containers."""
        f = tmp_path / "nostreamdur.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert result.valid
        assert result.av_drift == 0.0

    def test_av_drift_fields_populated_on_success(self, tmp_path):
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        payload = {
            "format": {"duration": "60.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "60.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "60.05"},
            ],
        }

        with patch("ffmpeg_longrun.validate.subprocess.run") as run:
            run.return_value = _probe(payload=payload)
            result = validate_video(f)

        assert result.valid
        assert result.video_duration > 0.0
        assert result.audio_duration > 0.0
        assert result.av_drift == pytest.approx(0.05, abs=0.01)


@pytest.mark.skipif(not _HAVE_FFPROBE, reason="ffprobe not installed")
def test_real_encode_round_trip(tmp_path):
    """End-to-end against the real binaries: generate a clip, then validate it.

    Skipped when ffmpeg/ffprobe are absent. Uses ffmpeg's synthetic sources —
    no media file is committed to this repository.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    clip = tmp_path / "synthetic.mp4"
    generated = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # The subject of this test is validate_video, not the local ffmpeg build.
    # An ffmpeg on PATH that cannot produce the fixture — no libx264, no aac,
    # a broken shared-library link — is a skip, not a failure. Only assert once
    # there is a real file to assert about.
    if generated.returncode != 0 or not clip.exists() or clip.stat().st_size == 0:
        pytest.skip(
            "local ffmpeg could not produce the H.264/AAC fixture "
            f"(rc={generated.returncode}): {generated.stderr[-200:]!r}"
        )

    result = validate_video(clip, expected_duration=2.0, duration_tolerance=0.15)
    assert result.valid, result.reason
    assert result.has_video
    assert result.has_audio
    assert result.duration == pytest.approx(2.0, abs=0.3)
