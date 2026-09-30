"""Tests for ffmpeg_longrun.runner.

Covers stall detection, progress parsing, stdout draining, and cleanup of
partial output. The subprocess under test is a short ``python3 -c`` script
rather than real ffmpeg, so the suite needs no media and no GPU.
"""

import io
import textwrap
from unittest.mock import patch

import pytest

from ffmpeg_longrun.runner import (
    _parse_time_to_seconds,
    _stall_kill_reason,
    _StdoutCapture,
    check_disk_space,
    run_ffmpeg_encode,
    run_ffmpeg_quick,
)


class _ScriptedClock:
    """Deterministic clock: returns successive values, clamping to the last.

    Lets a test drive stall and timeout decisions explicitly instead of
    depending on real elapsed time (which flakes under parallel or loaded
    runs).
    """

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self):
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


class _FakeStderr:
    """A file-like stderr yielding pre-scripted lines, then "" (EOF)."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0

    def readline(self):
        if self._i < len(self._lines):
            line = self._lines[self._i]
            self._i += 1
            return line
        return ""


class _FakeStdout:
    def read(self, n=-1):
        return ""  # no stdout — EOF immediately


class _FakeProc:
    """Minimal subprocess.Popen stand-in with scripted stderr and poll results.

    Combined with a _ScriptedClock this makes moov and stall behaviour fully
    deterministic: no real subprocess, no real sleeps, no wall-clock timing.
    """

    def __init__(self, stderr_lines, poll_returns):
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr(stderr_lines)
        self._poll = list(poll_returns)
        self._pi = 0
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        v = self._poll[min(self._pi, len(self._poll) - 1)]
        self._pi += 1
        if v is not None:
            self.returncode = v
        return v

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class TestStdoutCapture:
    def test_partial_trim_keeps_newest_valid_utf8_within_limit(self):
        cap = _StdoutCapture(max_bytes=10)
        cap._accept("a" * 8)
        cap._accept("\u00e9" * 4)  # 8 more bytes; oldest chunk must be cut partway
        assert cap.truncated is True
        assert cap.bytes_seen == 16
        assert len(cap.text.encode()) <= 10
        assert cap.text.endswith("\u00e9" * 4)

    def test_zero_limit_retains_nothing_but_counts(self):
        cap = _StdoutCapture(max_bytes=0)
        cap._accept("abc")
        assert cap.text == ""
        assert cap.bytes_seen == 3
        assert cap.truncated is True


# ── Time parsing ──────────────────────────────────────────────────────


class TestParseTimeToSeconds:
    def test_hms_format(self):
        assert _parse_time_to_seconds("01:30:45.50") == pytest.approx(5445.5)

    def test_ms_format(self):
        assert _parse_time_to_seconds("45:30.00") == pytest.approx(2730.0)

    def test_seconds_only(self):
        assert _parse_time_to_seconds("123.45") == pytest.approx(123.45)

    def test_empty_string(self):
        assert _parse_time_to_seconds("") is None

    def test_invalid(self):
        assert _parse_time_to_seconds("not-a-time") is None


# ── run_ffmpeg_encode ─────────────────────────────────────────────────


class TestRunFfmpegEncode:
    """Tests for the Popen-based long encode runner."""

    def test_successful_encode(self, tmp_path):
        """A successful encode returns FFmpegResult(success=True)."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake video data")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys, time
            for i in range(3):
                line = (
                    f"frame=  {i*100} fps= 60.0 size=   1024kB "
                    f"time=00:00:{i*10:02d}.00 speed=1.0x"
                )
                print(line, file=sys.stderr, flush=True)
                time.sleep(0.1)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test encode",
            stall_timeout=5,
            log_interval=1,
        )
        assert result.success
        assert result.returncode == 0
        assert result.killed_reason is None

    def test_stall_detection_kills_process(self, tmp_path):
        """Process killed after stall_timeout seconds of no progress output.

        Deterministic: the injected clock jumps past the stall window on the
        first loop reading, so the kill is instant and does not depend on real
        elapsed time.
        """
        output = tmp_path / "out.mp4"

        # Process that outputs nothing and would otherwise hang.
        cmd = ["python3", "-c", "import time; time.sleep(60)"]

        # start=0, then the first loop reading is 5s: no progress since start
        # (5s > 2s stall window) but within the 30s max ceiling → "stall".
        clock = _ScriptedClock([0.0, 5.0])
        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test stall",
            stall_timeout=2,
            max_timeout=30,
            time_source=clock,
        )
        assert not result.success
        assert result.killed_reason == "stall"
        assert not output.exists()

    def test_stall_resets_on_progress(self):
        """Stall timer resets when progress output is received.

        The window is measured from the LAST observed progress, not from
        encode start — so continuous progress keeps an encode alive no matter
        how long it runs, while a genuine gap still fires.
        """
        # Progress seen recently (t=95) though the encode started at t=0 and
        # total elapsed (100s) dwarfs the 10s stall window.
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=95.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            is None
        )
        # No progress since t=80 → 20s gap > 10s window → stall fires.
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=80.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            == "stall"
        )
        # Boundary: a gap exactly equal to the window does not fire (strict >).
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=90.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            is None
        )

    def test_max_timeout_takes_precedence_over_stall(self):
        """When both ceilings are breached at once, max_timeout wins."""
        assert (
            _stall_kill_reason(
                now=200.0,
                start_time=0.0,
                last_progress_time=0.0,
                max_timeout=100.0,
                stall_timeout=10.0,
            )
            == "max_timeout"
        )

    def test_partial_file_deleted_on_failure(self, tmp_path):
        """Output file is deleted when ffmpeg returns non-zero."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial corrupt data")

        cmd = ["python3", "-c", "import sys; sys.exit(1)"]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test cleanup",
            stall_timeout=5,
        )
        assert not result.success
        assert not output.exists(), "Partial output should be deleted on failure"

    def test_unlaunchable_command_returns_failure(self, tmp_path):
        """A command that cannot be launched fails cleanly instead of raising."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial")

        result = run_ffmpeg_encode(
            ["definitely-not-a-real-binary-xyz"],
            output,
            expected_duration=30.0,
            description="test unlaunchable",
        )
        assert not result.success
        assert result.returncode == -1
        assert result.stderr_tail
        assert not output.exists()

    def test_max_timeout_kills_process(self, tmp_path):
        """Process killed after max_timeout even if progress is flowing."""
        output = tmp_path / "out.mp4"

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys, time
            i = 0
            while True:
                print(f"frame=  {i} fps= 60.0 size=   1024kB time=00:00:{i:02d}.00 speed=1.0x",
                      file=sys.stderr, flush=True)
                time.sleep(0.5)
                i += 1
        """),
        ]

        # First loop reading jumps to 1000s, past the 3s max ceiling. Because
        # max is checked before stall (and before draining any progress),
        # "max_timeout" wins.
        clock = _ScriptedClock([0.0, 1000.0])
        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=3600.0,
            description="test max timeout",
            stall_timeout=10,
            max_timeout=3,
            time_source=clock,
        )
        assert not result.success
        assert result.killed_reason == "max_timeout"
        assert not output.exists()

    def test_progress_parsing_hardware_encoder_format(self, tmp_path):
        """Parses a hardware-encoder-style progress line correctly."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            line = (
                "frame=  3600 fps=142.3 q=21.0 size=  524288kB "
                "time=00:45:30.00 bitrate=1572.1kbits/s speed=1.2x"
            )
            print(line, file=sys.stderr, flush=True)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=4110.0,
            description="test progress parse",
            stall_timeout=5,
            log_interval=1,
        )
        assert result.success
        assert result.last_speed == pytest.approx(1.2)

    def test_progress_callback_called(self, tmp_path):
        """Progress callback is called with parsed values."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")
        callback_calls = []

        def cb(current, total, speed, fps):
            callback_calls.append((current, total, speed, fps))

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            print("frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=2.0x",
                  file=sys.stderr, flush=True)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=60.0,
            description="test callback",
            stall_timeout=5,
            progress_callback=cb,
        )
        assert result.success
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == pytest.approx(10.0)

    def test_progress_callback_exception_does_not_fail_encode(self, tmp_path):
        """A broken progress callback must not kill an otherwise good encode."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        def cb(*_args):
            raise ValueError("caller's meter is broken")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            print("frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=2.0x",
                  file=sys.stderr, flush=True)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=60.0,
            description="test broken callback",
            stall_timeout=5,
            progress_callback=cb,
        )
        assert result.success

    def test_large_stdout_does_not_deadlock_and_is_captured(self, tmp_path):
        """stdout is drained, so a child writing more than the pipe buffer
        (about 64 KB on Linux) does not deadlock. Captured stdout is exposed
        on FFmpegResult.stdout for callers that pipe JSON through it.

        Without the drainer this test hangs until the stall timeout kills the
        child — it would return killed_reason='stall' instead of succeeding.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        # Write 256 KB to stdout — 4x the typical Linux pipe buffer. Also emit
        # one progress line on stderr so the stall detector sees activity,
        # proving the assertion is about the stdout drain path.
        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            sys.stderr.write("frame=  100 fps= 60.0 size=   1024kB time=00:00:10.00 speed=1.0x\\n")
            sys.stderr.flush()
            payload = ("A" * 1024 + "\\n") * 256  # 256 KB
            sys.stdout.write(payload)
            sys.stdout.flush()
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test stdout drain",
            stall_timeout=10,
            max_timeout=30,
            log_interval=1,
        )
        assert result.success, f"expected success, got killed_reason={result.killed_reason}"
        assert result.killed_reason is None
        assert len(result.stdout) >= 256 * 1024
        assert result.stdout.startswith("A" * 1024)

    def test_empty_stdout_default(self, tmp_path):
        """Common case: ffmpeg writes media to its output argument and emits
        nothing on stdout. FFmpegResult.stdout is the empty string, not None.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            sys.stderr.write("frame=  100 fps= 60.0 size=   1024kB time=00:00:10.00 speed=1.0x\\n")
            sys.stderr.flush()
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test empty stdout",
            stall_timeout=5,
        )
        assert result.success
        assert result.stdout == ""

    # ── Bounded stdout capture (FF-1) ──────────────────────────────────

    _PROGRESS = 'sys.stderr.write("frame=  1 fps= 1 size=   1kB time=00:00:01.00 speed=1.0x\\n")'

    def _run_stdout_child(self, tmp_path, body, **kwargs):
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")
        script = "import sys\n" + self._PROGRESS + "\n" + textwrap.dedent(body)
        return run_ffmpeg_encode(
            ["python3", "-c", script],
            output,
            expected_duration=30.0,
            description="test stdout bounds",
            stall_timeout=20,
            max_timeout=60,
            **kwargs,
        )

    def test_high_volume_stdout_is_bounded_and_reports_truncation(self, tmp_path):
        """5 MiB on stdout retains at most the limit (the newest bytes), never
        deadlocks, and says that it truncated and how much it saw."""
        limit = 64 * 1024
        result = self._run_stdout_child(
            tmp_path,
            """\
            for i in range(5 * 1024):
                sys.stdout.write(f"{i:06d}" + "x" * 1017 + "\\n")
            sys.stdout.write("LAST-LINE\\n")
            sys.stdout.flush()
            """,
            max_stdout_bytes=limit,
        )
        assert result.success, f"killed_reason={result.killed_reason}"
        assert len(result.stdout.encode()) <= limit
        assert result.stdout_truncated is True
        assert result.stdout_bytes == 5 * 1024 * 1024 + len("LAST-LINE\n")
        assert result.stdout.endswith("LAST-LINE\n")

    def test_stdout_under_limit_is_not_flagged_truncated(self, tmp_path):
        result = self._run_stdout_child(tmp_path, 'sys.stdout.write("hello")\n')
        assert result.stdout == "hello"
        assert result.stdout_bytes == 5
        assert result.stdout_truncated is False

    def test_stdout_sink_receives_everything_and_nothing_is_retained(self, tmp_path):
        """With an explicit sink, stdout is streamed in full (nothing dropped)
        and not buffered in memory on the result."""
        sink = io.StringIO()
        result = self._run_stdout_child(
            tmp_path,
            """\
            for i in range(512):
                sys.stdout.write(("%04d" % i) * 256 + "\\n")
            """,
            stdout_sink=sink,
            max_stdout_bytes=1024,
        )
        assert result.success
        assert len(sink.getvalue()) == 512 * (1024 + 1)
        assert sink.getvalue().startswith("0000" * 256)
        assert result.stdout == ""
        assert result.stdout_bytes == 512 * (1024 + 1)
        assert result.stdout_truncated is False
        assert result.stdout_sink_error is None

    def test_stdout_sink_may_be_a_callable(self, tmp_path):
        chunks = []
        result = self._run_stdout_child(
            tmp_path, 'sys.stdout.write("abc" * 100)\n', stdout_sink=chunks.append
        )
        assert result.success
        assert "".join(chunks) == "abc" * 100

    def test_failing_sink_does_not_deadlock_and_is_reported(self, tmp_path):
        def bad_sink(_chunk):
            raise OSError("disk full")

        result = self._run_stdout_child(
            tmp_path,
            """\
            sys.stdout.write("z" * (1024 * 1024))
            """,
            stdout_sink=bad_sink,
        )
        assert result.success, f"killed_reason={result.killed_reason}"
        assert result.stdout_bytes == 1024 * 1024
        assert result.stdout_sink_error is not None
        assert "disk full" in result.stdout_sink_error

    def test_undecodable_stdout_does_not_stop_the_drain(self, tmp_path):
        """Invalid UTF-8 must not kill the drain thread (which would refill the
        pipe and hang the child until the stall kill)."""
        result = self._run_stdout_child(
            tmp_path,
            """\
            sys.stdout.buffer.write(b"\\xff\\xfe" * 200_000)
            sys.stdout.buffer.flush()
            """,
        )
        assert result.success, f"killed_reason={result.killed_reason}"
        assert result.stdout_bytes > 0

    def test_negative_stdout_limit_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_stdout_bytes"):
            run_ffmpeg_encode(
                ["python3", "-c", "pass"],
                tmp_path / "out.mp4",
                expected_duration=0,
                max_stdout_bytes=-1,
            )

    def test_moov_rewrite_extends_stall(self, tmp_path):
        """The moov atom rewrite phase extends the stall timeout.

        Progress, then a 'moving the moov atom' line, are consumed while the
        clock reads 0; the clock then jumps to 100s (far past the 2s base
        window) with the process still alive. The encode survives ONLY because
        the moov line raised the effective stall window.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        stderr_lines = [
            "frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=1.0x\n",
            "Starting second pass: moving the moov atom to the beginning of the file\n",
        ]
        # poll: alive while the lines are consumed and the clock is still 0,
        # then exits cleanly (0) on the iteration where the clock reads 100.
        fake = _FakeProc(stderr_lines=stderr_lines, poll_returns=[None, None, None, 0])
        clock = _ScriptedClock([0.0, 0.0, 0.0, 0.0, 100.0])

        with patch("ffmpeg_longrun.runner.subprocess.Popen", return_value=fake):
            result = run_ffmpeg_encode(
                ["ffmpeg", "-i", "in.mp4", str(output)],
                output,
                expected_duration=10.0,
                description="test moov",
                stall_timeout=2,
                max_timeout=1000,
                time_source=clock,
            )

        assert result.success
        assert result.killed_reason is None
        # Never killed → the moov extension applied.
        assert fake.terminated is False
        assert fake.killed is False


# ── run_ffmpeg_quick ──────────────────────────────────────────────────


class TestRunFfmpegQuick:
    """Tests for the subprocess.run wrapper."""

    def test_success(self):
        """Successful command returns CompletedProcess."""
        result = run_ffmpeg_quick(
            ["python3", "-c", "print('hello')"],
            description="test success",
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_timeout_deletes_output(self, tmp_path):
        """TimeoutExpired deletes the output file."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial data")

        with pytest.raises(RuntimeError, match="timed out"):
            run_ffmpeg_quick(
                ["python3", "-c", "import time; time.sleep(60)"],
                output_path=output,
                timeout=1,
                description="test timeout",
            )
        assert not output.exists(), "Output should be deleted on timeout"

    def test_nonzero_exit_deletes_output(self, tmp_path):
        """Non-zero exit code deletes the output file."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial data")

        with pytest.raises(RuntimeError, match="failed"):
            run_ffmpeg_quick(
                ["python3", "-c", "import sys; sys.exit(1)"],
                output_path=output,
                timeout=5,
                description="test exit",
            )
        assert not output.exists(), "Output should be deleted on non-zero exit"

    def test_no_output_path_no_crash(self):
        """Non-zero exit without output_path doesn't crash."""
        with pytest.raises(RuntimeError):
            run_ffmpeg_quick(
                ["python3", "-c", "import sys; sys.exit(1)"],
                description="test no output",
            )


# ── check_disk_space ──────────────────────────────────────────────────


class TestCheckDiskSpace:
    def test_passes_when_space_available(self, tmp_path):
        """One byte is always available; the check returns silently."""
        assert check_disk_space(tmp_path, 1) is None

    def test_raises_when_space_insufficient(self, tmp_path):
        """An absurd requirement raises with a message naming both figures."""
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            check_disk_space(tmp_path, 10**18)

    def test_walks_up_to_an_existing_ancestor(self, tmp_path):
        """A not-yet-created output directory is checked against its parent."""
        missing = tmp_path / "does" / "not" / "exist" / "yet"
        assert check_disk_space(missing, 1) is None
