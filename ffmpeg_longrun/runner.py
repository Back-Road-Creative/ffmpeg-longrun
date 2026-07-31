"""Subprocess execution for FFmpeg jobs that run for hours.

Two entry points:

``run_ffmpeg_encode``
    Long encodes. Uses ``Popen`` with threaded pipe readers, parses ffmpeg's
    progress output, kills a process that has stopped making progress, and
    deletes the partial output on every failure path.

``run_ffmpeg_quick``
    Short operations — stream copies, probes, audio measurement. A thin
    ``subprocess.run`` wrapper with a timeout and the same cleanup guarantee.

The design problem these solve is that a plain ``subprocess.run(..., timeout=N)``
around a multi-hour encode forces you to guess N. Guess low and you kill
healthy encodes on a loaded machine; guess high and a genuinely hung ffmpeg
holds the job for hours. ``run_ffmpeg_encode`` measures *progress* instead of
total elapsed time, so a job that keeps reporting frames survives indefinitely
and a job that stops reporting dies in minutes.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

logger = logging.getLogger(__name__)

__all__ = [
    "FFmpegResult",
    "run_ffmpeg_encode",
    "run_ffmpeg_quick",
    "check_disk_space",
]

#: Wall-clock gap with no progress output that counts as a stall, in seconds.
DEFAULT_STALL_TIMEOUT = 600

#: Absolute ceiling on a single encode, in seconds. A backstop, not the gate.
DEFAULT_MAX_TIMEOUT = 86_400

#: Stall window used while ffmpeg relocates the moov atom (see below).
MOOV_REWRITE_STALL_TIMEOUT = 1_800


@dataclass
class FFmpegResult:
    """Outcome of a long-running FFmpeg encode.

    Attributes
    ----------
    success:
        ``True`` only when ffmpeg exited 0 and was not killed.
    returncode:
        Process exit code, or ``-1`` when the process could not be started.
    wall_time:
        Seconds from launch to exit, measured on the injected clock.
    stderr_tail:
        Last ~1500 characters of stderr. ffmpeg puts everything useful there.
    killed_reason:
        ``"stall"``, ``"max_timeout"``, or ``None`` if the process exited on
        its own (whether successfully or not).
    last_progress_time:
        Seconds from launch to the last progress line observed.
    last_speed:
        Last ``speed=Nx`` value parsed from the progress output.
    stdout:
        Full captured stdout. Empty in the common case — ffmpeg writes encoded
        media to its output argument and says nothing on stdout. Populated
        when a caller pipes output through it (``-progress pipe:1``,
        ``loudnorm=print_format=json``, ``ffprobe -of json``). stdout is
        drained in a background thread whether or not the caller wants it, so
        the OS pipe buffer cannot fill and deadlock the child.
    """

    success: bool
    returncode: int
    wall_time: float
    stderr_tail: str
    killed_reason: str | None = None
    last_progress_time: float = 0.0
    last_speed: float = 0.0
    stdout: str = ""


def _parse_time_to_seconds(time_str: str) -> float | None:
    """Parse an ffmpeg time string (``HH:MM:SS.ss``, ``MM:SS.ss``, ``SS.ss``)."""
    time_str = time_str.strip()
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return None


def _stderr_reader(pipe, queue: Queue) -> None:
    """Thread target: read stderr lines into a queue, then post a sentinel."""
    try:
        for line in iter(pipe.readline, ""):
            queue.put(line)
    except (ValueError, OSError):
        pass
    finally:
        queue.put(None)  # Sentinel


def _stdout_drain(pipe, buffer: list[str]) -> None:
    """Thread target: drain stdout into *buffer*.

    ffmpeg normally writes encoded media to its output argument and emits
    nothing on stdout, but several flags do emit there (``-progress pipe:1``,
    ``loudnorm print_format=json``, ``ffprobe -of json``). With
    ``stdout=PIPE`` and nothing reading it, the OS pipe buffer (about 64 KB on
    Linux) fills and blocks the child mid-write. That surfaces as a silent
    hang which the stall detector eventually kills with no useful diagnostic —
    the encode looks stuck, and the real cause is a full pipe.

    Draining unconditionally costs one background thread and an append-only
    list, and makes that failure mode unrepresentable.

    Chunk reads rather than line reads: ``-progress pipe:1`` emits newline
    terminated ``k=v`` records, but ``ffprobe -of json`` emits one large blob
    with no trailing newline. Chunks handle both.
    """
    try:
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                break
            buffer.append(chunk)
    except (ValueError, OSError):
        pass


def _stall_kill_reason(
    now: float,
    start_time: float,
    last_progress_time: float,
    max_timeout: float,
    stall_timeout: float,
) -> str | None:
    """Decide whether the encode should be killed at clock reading *now*.

    A pure function: it takes the clock reading as an argument and performs no
    I/O, so stall and timeout behaviour is deterministically testable with an
    explicit clock instead of real elapsed wall time (which flakes under CPU
    contention).

    Returns ``"max_timeout"`` when the absolute wall-time ceiling is exceeded
    (checked first), ``"stall"`` when no progress has been observed within
    *stall_timeout* seconds, else ``None``. The stall window is measured from
    the LAST observed progress, so continuous progress keeps an encode alive
    regardless of total elapsed time.
    """
    if now - start_time > max_timeout:
        return "max_timeout"
    if now - last_progress_time > stall_timeout:
        return "stall"
    return None


def run_ffmpeg_encode(
    cmd: list[str],
    output_path: Path,
    expected_duration: float,
    description: str = "ffmpeg encode",
    stall_timeout: int = DEFAULT_STALL_TIMEOUT,
    max_timeout: int = DEFAULT_MAX_TIMEOUT,
    progress_callback: Callable[[float, float, float, float], None] | None = None,
    log_interval: int = 30,
    time_source: Callable[[], float] = time.monotonic,
) -> FFmpegResult:
    """Run a long FFmpeg encode with progress monitoring and stall detection.

    Launches *cmd* under ``Popen``, reads stderr on a background thread, and
    kills the process if no progress line appears for *stall_timeout* seconds.
    Deletes *output_path* on every failure path so a half-written file can
    never be mistaken for a finished one by a later stage.

    Parameters
    ----------
    cmd:
        The full ffmpeg command as a list of strings.
    output_path:
        The file ffmpeg is writing. Deleted on failure.
    expected_duration:
        Expected output duration in seconds, used to compute progress percent
        and ETA. Pass ``0`` to disable those log lines.
    description:
        Human-readable label used in log messages.
    stall_timeout:
        Kill after this many seconds with no progress output. The window
        resets on every progress line.
    max_timeout:
        Absolute wall-time ceiling. A backstop for a process that keeps
        emitting progress but will never finish.
    progress_callback:
        Called as ``callback(current_secs, total_secs, speed, fps)`` on each
        parsed progress line. Exceptions raised by the callback are swallowed
        — a broken progress meter must not kill a six-hour encode.
    log_interval:
        Minimum seconds between progress log lines.
    time_source:
        Monotonic clock used for stall and timeout decisions. Injectable so
        tests can drive time explicitly.

    Returns
    -------
    FFmpegResult
    """
    output_path = Path(output_path)
    start_time = time_source()
    last_progress_time = start_time
    last_log_time = start_time
    last_speed = 0.0
    last_fps = 0.0
    current_secs = 0.0
    stderr_lines: list[str] = []
    killed_reason = None
    in_moov_rewrite = False
    effective_stall_timeout = stall_timeout

    logger.info("  %s: starting (stall=%ss, max=%ss)", description, stall_timeout, max_timeout)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        _cleanup_output(output_path)
        return FFmpegResult(
            success=False,
            returncode=-1,
            wall_time=0.0,
            stderr_tail=str(e),
            killed_reason=None,
        )

    stderr_queue: Queue = Queue()
    reader_thread = threading.Thread(target=_stderr_reader, args=(proc.stderr, stderr_queue))
    reader_thread.daemon = True
    reader_thread.start()

    stdout_chunks: list[str] = []
    stdout_thread = threading.Thread(target=_stdout_drain, args=(proc.stdout, stdout_chunks))
    stdout_thread.daemon = True
    stdout_thread.start()

    try:
        while True:
            now = time_source()

            kill_reason = _stall_kill_reason(
                now, start_time, last_progress_time, max_timeout, effective_stall_timeout
            )

            if kill_reason == "max_timeout":
                killed_reason = "max_timeout"
                logger.error("  %s: max timeout (%ss) reached, killing", description, max_timeout)
                _terminate(proc)
                break

            if kill_reason == "stall":
                killed_reason = "stall"
                elapsed = now - last_progress_time
                logger.error(
                    "  %s: stalled for %.0fs (limit %ss), killing",
                    description,
                    elapsed,
                    effective_stall_timeout,
                )
                _terminate(proc)
                break

            got_output = False
            while True:
                try:
                    line = stderr_queue.get_nowait()
                except Empty:
                    break

                if line is None:
                    # Reader thread finished — the process is done or closing.
                    got_output = True
                    break

                got_output = True
                stderr_lines.append(line)
                if len(stderr_lines) > 200:
                    stderr_lines = stderr_lines[-200:]

                # Relocating the moov atom to the front of the file
                # (``-movflags +faststart``) is a second pass over the finished
                # output. It reports no frame progress at all, so an untouched
                # stall detector kills a perfectly healthy job right at the
                # finish line — on a large 4K file this pass alone can run for
                # many minutes. Widen the window while it is happening and put
                # it back the moment frame progress resumes.
                if "Starting second pass" in line or "moving the moov atom" in line.lower():
                    if not in_moov_rewrite:
                        in_moov_rewrite = True
                        effective_stall_timeout = max(stall_timeout, MOOV_REWRITE_STALL_TIMEOUT)
                        logger.warning(
                            "  %s: moov atom rewrite detected; extending stall timeout "
                            "from %ss to %ss — stall detection is suppressed until "
                            "frame progress resumes",
                            description,
                            stall_timeout,
                            effective_stall_timeout,
                        )

                # ffmpeg progress lines look like:
                #   frame= 1234 fps= 142 ... time=00:45:30.00 ... speed=1.2x
                if "time=" in line and ("frame=" in line or "size=" in line):
                    last_progress_time = now

                    if in_moov_rewrite and "frame=" in line:
                        in_moov_rewrite = False
                        effective_stall_timeout = stall_timeout

                    time_match = re.search(r"time=(\S+)", line)
                    if time_match:
                        parsed = _parse_time_to_seconds(time_match.group(1))
                        if parsed is not None:
                            current_secs = parsed

                    speed_match = re.search(r"speed=\s*([\d.]+)x", line)
                    if speed_match:
                        last_speed = float(speed_match.group(1))

                    fps_match = re.search(r"fps=\s*([\d.]+)", line)
                    if fps_match:
                        last_fps = float(fps_match.group(1))

                    if progress_callback:
                        try:
                            progress_callback(current_secs, expected_duration, last_speed, last_fps)
                        except Exception:
                            pass

                    if now - last_log_time >= log_interval and expected_duration > 0:
                        pct = current_secs / expected_duration * 100
                        remaining_secs = (
                            (expected_duration - current_secs) / last_speed if last_speed > 0 else 0
                        )
                        eta_min = remaining_secs / 60 if remaining_secs > 0 else 0

                        cur_m, cur_s = divmod(int(current_secs), 60)
                        tot_m, tot_s = divmod(int(expected_duration), 60)
                        logger.info(
                            "  Progress: %d:%02d / %d:%02d (%.1f%%) — %.0ffps, %.1fx, ETA ~%.0fmin",
                            cur_m,
                            cur_s,
                            tot_m,
                            tot_s,
                            pct,
                            last_fps,
                            last_speed,
                            eta_min,
                        )
                        last_log_time = now

            retcode = proc.poll()
            if retcode is not None:
                while True:
                    try:
                        line = stderr_queue.get(timeout=1)
                    except Empty:
                        break
                    if line is None:
                        break
                    stderr_lines.append(line)
                break

            if not got_output:
                time.sleep(0.25)

    except Exception as e:
        logger.error("  %s: unexpected error: %s", description, e)
        _terminate(proc)
        _cleanup_output(output_path)
        return FFmpegResult(
            success=False,
            returncode=-1,
            wall_time=time_source() - start_time,
            stderr_tail=str(e),
        )

    wall_time = time_source() - start_time
    returncode = proc.returncode if proc.returncode is not None else -1
    stderr_tail = "".join(stderr_lines[-50:])

    success = returncode == 0 and killed_reason is None

    if not success:
        _cleanup_output(output_path)
        if killed_reason:
            logger.error("  %s: killed (%s) after %.0fs", description, killed_reason, wall_time)
        else:
            logger.error("  %s: failed with returncode %s", description, returncode)
            if stderr_tail:
                logger.error("  stderr tail:\n%s", stderr_tail[-1500:])

    reader_thread.join(timeout=5)
    stdout_thread.join(timeout=5)
    captured_stdout = "".join(stdout_chunks)

    return FFmpegResult(
        success=success,
        returncode=returncode,
        wall_time=wall_time,
        stderr_tail=stderr_tail[-1500:],
        killed_reason=killed_reason,
        last_progress_time=last_progress_time - start_time,
        last_speed=last_speed,
        stdout=captured_stdout,
    )


def run_ffmpeg_quick(
    cmd: list[str],
    output_path: Path | None = None,
    timeout: int = 300,
    description: str = "ffmpeg",
) -> subprocess.CompletedProcess:
    """Run a short FFmpeg or ffprobe command with a timeout and cleanup.

    On timeout or a non-zero exit, deletes *output_path* if given and raises
    ``RuntimeError`` carrying the tail of stderr.

    Parameters
    ----------
    cmd:
        Command as a list of strings.
    output_path:
        Optional output file to delete on failure.
    timeout:
        Hard timeout in seconds. Appropriate here precisely because these
        operations are short — for a long encode use
        :func:`run_ffmpeg_encode` instead.
    description:
        Human-readable label used in error messages.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    RuntimeError
        On timeout or a non-zero exit code.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if output_path:
            _cleanup_output(Path(output_path))
        raise RuntimeError(
            f"{description}: timed out after {timeout}s. Command: {' '.join(cmd[:6])}..."
        ) from exc

    if result.returncode != 0:
        if output_path:
            _cleanup_output(Path(output_path))
        stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
        raise RuntimeError(f"{description} failed (rc={result.returncode}):\n{stderr_tail}")

    return result


def check_disk_space(path: Path, required_bytes: int) -> None:
    """Raise ``RuntimeError`` if *path*'s filesystem has under *required_bytes* free.

    A preflight check for any large encode or concat: fail immediately with a
    clear message rather than run for hours and produce a truncated file that
    has to be restarted from scratch. Walks up to the nearest existing
    ancestor, so it works before the output directory has been created.
    """
    target = Path(path)
    while not target.exists():
        parent = target.parent
        if parent == target:  # reached the filesystem root
            break
        target = parent
    usage = shutil.disk_usage(target)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {path}: "
            f"{usage.free / 1_073_741_824:.1f} GB free, "
            f"need ~{required_bytes / 1_073_741_824:.1f} GB"
        )


def _terminate(proc) -> None:
    """SIGTERM the process, escalating to SIGKILL if it does not go quietly."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _cleanup_output(output_path: Path) -> None:
    """Delete a partial or corrupt output file if it exists."""
    try:
        if output_path.exists():
            size = output_path.stat().st_size
            output_path.unlink()
            logger.warning("  Cleaned up partial output: %s (%d bytes)", output_path.name, size)
    except OSError as e:
        logger.warning("  Failed to clean up %s: %s", output_path, e)
