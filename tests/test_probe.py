"""FFprobeClient retries transient timeouts and fails fast on corruption.

A valid file probed while a heavy concurrent encode pins the CPU can exceed
the ffprobe timeout and raise ``subprocess.TimeoutExpired``. Treating that as
corruption is a false failure that can abort a long job. The client retries a
timeout a few times before raising, while non-timeout errors stay fatal
immediately.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from ffmpeg_longrun.probe import (
    FFPROBE_TIMEOUT_RETRIES,
    FFprobeClient,
    FFprobeError,
)


def _ok_result(stdout: str = '{"format": {"duration": "12.5"}}'):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _timeout(*_args, **_kwargs):
    raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Keep the backoff from slowing the suite; the retry path still exercises.
    monkeypatch.setattr("ffmpeg_longrun.probe.time.sleep", lambda _s: None)


def test_timeout_twice_then_success_returns_value():
    """Times out twice, succeeds on the third attempt → returns the value."""
    sub = MagicMock()
    sub.run.side_effect = [
        subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
        subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
        _ok_result(),
    ]

    client = FFprobeClient(sub)
    data = client.probe("/some/file.mp4")

    assert data == {"format": {"duration": "12.5"}}
    assert sub.run.call_count == 3  # retried, did not give up on first timeout


def test_timeout_every_attempt_raises_after_n_retries():
    """Times out on every attempt → raises TimeoutExpired after N attempts."""
    sub = MagicMock()
    sub.run.side_effect = _timeout

    client = FFprobeClient(sub)
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        client.probe("/some/file.mp4")

    # Raised only after exhausting all retries, not on the first timeout.
    assert sub.run.call_count == FFPROBE_TIMEOUT_RETRIES
    # Message makes clear it was timeout-after-retries, not corruption.
    assert "not corruption" in (exc_info.value.output or "")


def test_non_timeout_error_raises_immediately_without_retry():
    """A non-zero return code (genuine corruption) fails fast, no retry."""
    sub = MagicMock()
    sub.run.return_value = MagicMock(returncode=1, stdout="", stderr="moov atom not found")

    client = FFprobeClient(sub)
    with pytest.raises(FFprobeError):
        client.probe("/some/file.mp4")

    # Corruption is never retried — exactly one subprocess invocation.
    assert sub.run.call_count == 1


def test_subprocess_exception_is_not_retried():
    """A non-timeout exception from subprocess.run propagates immediately."""
    sub = MagicMock()
    sub.run.side_effect = OSError("ffprobe not found")

    client = FFprobeClient(sub)
    with pytest.raises(OSError):
        client.probe("/some/file.mp4")

    assert sub.run.call_count == 1


def test_unparseable_json_raises_ffprobe_error():
    """Garbage on stdout is a probe failure, not a stray JSONDecodeError."""
    sub = MagicMock()
    sub.run.return_value = _ok_result(stdout="}{ not json")

    client = FFprobeClient(sub)
    with pytest.raises(FFprobeError, match="unparseable"):
        client.probe("/some/file.mp4")


# ── Command construction ──────────────────────────────────────────────


def _captured_cmd(sub) -> list:
    return list(sub.run.call_args[0][0])


def test_default_command_shape():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    FFprobeClient(sub).probe("/clip.mp4")

    cmd = _captured_cmd(sub)
    assert cmd[0] == "ffprobe"
    assert cmd[1:3] == ["-v", "quiet"]
    assert "-show_format" in cmd
    assert "-show_streams" not in cmd
    assert cmd[-1] == "/clip.mp4"


def test_show_streams_and_verbose_level():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    FFprobeClient(sub).probe("/clip.mp4", show_streams=True, verbose_level="error")

    cmd = _captured_cmd(sub)
    assert cmd[1:3] == ["-v", "error"]
    assert "-show_streams" in cmd


def test_entries_replaces_format_and_streams():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    FFprobeClient(sub).probe("/clip.mp4", entries="format=duration")

    cmd = _captured_cmd(sub)
    assert "-show_entries" in cmd
    assert cmd[cmd.index("-show_entries") + 1] == "format=duration"
    assert "-show_format" not in cmd
    assert "-print_format" not in cmd


def test_custom_ffprobe_binary_is_used():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    FFprobeClient(sub, ffprobe_bin="/opt/tools/ffprobe").probe("/clip.mp4")

    assert _captured_cmd(sub)[0] == "/opt/tools/ffprobe"


def test_per_call_timeout_overrides_default():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    FFprobeClient(sub, timeout=30).probe("/clip.mp4", timeout=5)

    assert sub.run.call_args.kwargs["timeout"] == 5


# ── Convenience helpers ───────────────────────────────────────────────


def test_probe_duration_returns_value():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    assert FFprobeClient(sub).probe_duration("/clip.mp4") == pytest.approx(12.5)


def test_probe_duration_returns_zero_on_failure():
    sub = MagicMock()
    sub.run.return_value = MagicMock(returncode=1, stdout="", stderr="broken")

    assert FFprobeClient(sub).probe_duration("/clip.mp4") == 0.0


def test_probe_duration_returns_zero_when_field_absent():
    sub = MagicMock()
    sub.run.return_value = _ok_result(stdout="{}")

    assert FFprobeClient(sub).probe_duration("/clip.mp4") == 0.0


def test_probe_or_none_returns_none_on_failure():
    sub = MagicMock()
    sub.run.return_value = MagicMock(returncode=1, stdout="", stderr="broken")

    assert FFprobeClient(sub).probe_or_none("/clip.mp4") is None


def test_probe_or_none_returns_dict_on_success():
    sub = MagicMock()
    sub.run.return_value = _ok_result()

    assert FFprobeClient(sub).probe_or_none("/clip.mp4") == {"format": {"duration": "12.5"}}


def test_default_subprocess_module_is_the_real_one():
    """Constructing with no arguments uses the stdlib subprocess module."""
    client = FFprobeClient()
    assert client._subprocess is subprocess
