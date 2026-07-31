"""Tests for ffmpeg_longrun.encoders.

The NVENC probe is mocked throughout — the suite must pass on a machine with
no GPU, which is also the case the fallback exists for.
"""

from __future__ import annotations

import dataclasses
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ffmpeg_longrun.config import (
    DEFAULT_VIDEO_PROFILE,
    VERTICAL_SOCIAL_VIDEO_PROFILE,
    VideoProfile,
)
from ffmpeg_longrun.encoders import (
    NVENCUnavailableError,
    build_cpu_args,
    build_encode_args,
    build_nvenc_args,
    nvenc_available,
    require_nvenc,
)


def _pair(args: list[str], flag: str) -> str:
    """Return the value following *flag* in an ffmpeg argument list."""
    return args[args.index(flag) + 1]


# ── nvenc_available ───────────────────────────────────────────────────


class TestNvencAvailable:
    def test_true_when_probe_exits_zero(self):
        with patch("ffmpeg_longrun.encoders.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            assert nvenc_available() is True

    def test_false_when_probe_exits_nonzero(self):
        with patch("ffmpeg_longrun.encoders.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1)
            assert nvenc_available() is False

    def test_false_when_ffmpeg_is_missing(self):
        with patch("ffmpeg_longrun.encoders.subprocess.run", side_effect=FileNotFoundError):
            assert nvenc_available() is False

    def test_false_when_probe_times_out(self):
        with patch(
            "ffmpeg_longrun.encoders.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10),
        ):
            assert nvenc_available() is False

    def test_probe_uses_a_256px_test_source(self):
        """NVENC rejects frames below a driver minimum, so the probe cannot
        use a 1x1 source or it would fail for the wrong reason."""
        with patch("ffmpeg_longrun.encoders.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            nvenc_available()

        cmd = run.call_args[0][0]
        assert "testsrc=duration=1:size=256x256:rate=1" in cmd
        assert _pair(cmd, "-c:v") == "hevc_nvenc"

    def test_probe_honours_custom_codec_and_binary(self):
        with patch("ffmpeg_longrun.encoders.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            nvenc_available(codec="h264_nvenc", ffmpeg_bin="/opt/tools/ffmpeg")

        cmd = run.call_args[0][0]
        assert cmd[0] == "/opt/tools/ffmpeg"
        assert _pair(cmd, "-c:v") == "h264_nvenc"


class TestRequireNvenc:
    def test_returns_none_when_available(self):
        with patch("ffmpeg_longrun.encoders.nvenc_available", return_value=True):
            assert require_nvenc() is None

    def test_raises_when_unavailable(self):
        with patch("ffmpeg_longrun.encoders.nvenc_available", return_value=False):
            with pytest.raises(NVENCUnavailableError, match="NVIDIA GPU"):
                require_nvenc()

    def test_error_is_a_runtime_error(self):
        assert issubclass(NVENCUnavailableError, RuntimeError)


# ── Argument builders ─────────────────────────────────────────────────


class TestArgumentBuilders:
    def test_nvenc_args_carry_the_profile_quality_settings(self):
        args = build_nvenc_args()
        assert _pair(args, "-c:v") == DEFAULT_VIDEO_PROFILE.nvenc_codec
        assert _pair(args, "-preset") == "p7"
        assert _pair(args, "-cq") == "18"
        assert _pair(args, "-profile:v") == "main10"
        assert _pair(args, "-pix_fmt") == "p010le"
        assert _pair(args, "-rc") == "vbr"
        assert _pair(args, "-g") == "60"

    def test_cpu_args_carry_the_profile_quality_settings(self):
        args = build_cpu_args()
        assert _pair(args, "-c:v") == "libx265"
        assert _pair(args, "-preset") == "slow"
        assert _pair(args, "-crf") == "18"
        assert _pair(args, "-profile:v") == "main10"
        assert _pair(args, "-pix_fmt") == "yuv420p10le"

    def test_cpu_args_pin_the_keyframe_interval_on_both_ends(self):
        """Open GOP and a drifting keyint break clean segmenting downstream."""
        params = _pair(build_cpu_args(), "-x265-params")
        assert "keyint=60" in params
        assert "min-keyint=60" in params
        assert "open-gop=0" in params

    def test_nvenc_path_selected_when_available(self):
        args = build_encode_args(2160, has_nvenc=True)
        assert _pair(args, "-c:v") == "hevc_nvenc"

    def test_cpu_path_selected_when_unavailable(self):
        args = build_encode_args(2160, has_nvenc=False)
        assert _pair(args, "-c:v") == "libx265"

    def test_fallback_is_logged_as_a_warning(self, caplog):
        """A silent CPU fallback costs hours; it has to be visible in the log."""
        with caplog.at_level("WARNING", logger="ffmpeg_longrun.encoders"):
            build_encode_args(2160, has_nvenc=False)
        assert "CPU encode" in caplog.text

    def test_colour_metadata_is_tagged(self):
        args = build_encode_args(2160, has_nvenc=True)
        assert _pair(args, "-color_range") == "tv"
        assert _pair(args, "-colorspace") == "bt709"
        assert _pair(args, "-color_primaries") == "bt709"
        assert _pair(args, "-color_trc") == "bt709"

    def test_faststart_is_appended_by_default(self):
        args = build_encode_args(2160, has_nvenc=True)
        assert _pair(args, "-movflags") == "+faststart"

    def test_faststart_can_be_turned_off(self):
        profile = DEFAULT_VIDEO_PROFILE.replace(faststart=False)
        assert "-movflags" not in build_encode_args(2160, has_nvenc=True, profile=profile)


class TestBitrateFloors:
    def test_4k_floor(self):
        args = build_encode_args(2160, has_nvenc=True)
        assert _pair(args, "-b:v") == "40000k"
        assert _pair(args, "-maxrate") == "80000k"  # 2.0x
        assert _pair(args, "-bufsize") == "120000k"  # 3.0x

    def test_1080p_floor(self):
        args = build_encode_args(1080, has_nvenc=True)
        assert _pair(args, "-b:v") == "15000k"

    def test_sub_1080p_floor(self):
        args = build_encode_args(720, has_nvenc=True)
        assert _pair(args, "-b:v") == "5000k"

    def test_absolute_overrides_win_over_ratios(self):
        args = build_encode_args(1920, has_nvenc=True, profile=VERTICAL_SOCIAL_VIDEO_PROFILE)
        assert _pair(args, "-maxrate") == "8000k"
        assert _pair(args, "-bufsize") == "12000k"

    def test_profile_helper_reports_the_same_numbers(self):
        assert DEFAULT_VIDEO_PROFILE.rate_control_kbps(2160) == (40_000, 80_000, 120_000)
        assert DEFAULT_VIDEO_PROFILE.bitrate_floor_for_height(1080) == 15_000
        assert DEFAULT_VIDEO_PROFILE.bitrate_floor_for_height(480) == 5_000


class TestAudioModes:
    def test_copy_is_the_default(self):
        args = build_encode_args(2160, has_nvenc=True)
        assert _pair(args, "-c:a") == "copy"
        assert "-an" not in args

    def test_encode_uses_the_profile_codec_and_bitrate(self):
        args = build_encode_args(2160, has_nvenc=True, audio="encode")
        assert _pair(args, "-c:a") == "aac"
        assert _pair(args, "-b:a") == "384k"

    def test_none_strips_the_audio_stream(self):
        """Some delivery targets re-normalise quiet audio back up on their
        side; a stream that was never shipped cannot be re-normalised."""
        args = build_encode_args(1920, has_nvenc=True, audio="none")
        assert "-an" in args
        assert "-c:a" not in args
        assert "-b:a" not in args

    def test_vertical_profile_uses_a_lower_audio_bitrate(self):
        args = build_encode_args(
            1920, has_nvenc=True, audio="encode", profile=VERTICAL_SOCIAL_VIDEO_PROFILE
        )
        assert _pair(args, "-b:a") == "128k"

    def test_unknown_audio_mode_raises(self):
        with pytest.raises(ValueError, match="audio must be"):
            build_encode_args(2160, has_nvenc=True, audio="loudnorm")  # type: ignore[arg-type]


class TestProfileCustomisation:
    def test_replace_returns_a_new_frozen_profile(self):
        custom = DEFAULT_VIDEO_PROFILE.replace(nvenc_cq=23, cpu_crf=23)
        assert custom.nvenc_cq == 23
        assert DEFAULT_VIDEO_PROFILE.nvenc_cq == 18, "the default must not be mutated"
        assert _pair(build_encode_args(2160, has_nvenc=True, profile=custom), "-cq") == "23"

    def test_a_fully_custom_profile_flows_through(self):
        profile = VideoProfile(
            nvenc_codec="av1_nvenc",
            output_fps="25",
            gop_size=50,
            encoder_profile="main",
        )
        args = build_encode_args(1080, has_nvenc=True, profile=profile)
        assert _pair(args, "-c:v") == "av1_nvenc"
        assert _pair(args, "-r") == "25"
        assert _pair(args, "-g") == "50"
        assert _pair(args, "-profile:v") == "main"

    def test_profiles_are_immutable(self):
        """A shared module-level default must not be mutable in place."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_VIDEO_PROFILE.nvenc_cq = 30  # type: ignore[misc]
