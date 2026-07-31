"""A single parameterised ffprobe wrapper.

Codebases that shell out to ffprobe tend to accumulate a dozen near-identical
``subprocess.run(["ffprobe", ...])`` call sites, each with its own flags,
timeout, and error handling — and each with its own subtly different idea of
what a failure means. :class:`FFprobeClient` is one call site with the
differences expressed as arguments.

The one behaviour worth knowing about is the retry rule: a probe that *times
out* is retried, and a probe that *fails* is not. The distinction matters on a
loaded machine. A perfectly valid file probed while a heavy encode pins every
core can blow past a 30-second timeout; treating that as corruption aborts a
long job for no reason. A non-zero exit code or unparseable JSON, on the other
hand, is the file actually being broken, and retrying it just wastes time.
"""

from __future__ import annotations

import json
import subprocess
import time
import types
from pathlib import Path

__all__ = [
    "FFprobeClient",
    "FFprobeError",
    "FFPROBE_TIMEOUT_RETRIES",
    "FFPROBE_TIMEOUT_BACKOFF_SEC",
]

#: Total subprocess attempts made when ffprobe keeps timing out.
FFPROBE_TIMEOUT_RETRIES = 3

#: Base backoff in seconds, multiplied by the attempt number (linear backoff).
FFPROBE_TIMEOUT_BACKOFF_SEC = 2.0


class FFprobeError(Exception):
    """ffprobe returned a non-zero exit code, or emitted unparseable output."""


class FFprobeClient:
    """Configurable ffprobe wrapper with JSON output.

    Parameters
    ----------
    subprocess_module:
        The module used for ``run()``. Defaults to the standard
        :mod:`subprocess`. Injectable so a caller can substitute a fake in
        tests without patching a module attribute.
    timeout:
        Default per-call timeout in seconds.
    ffprobe_bin:
        ffprobe executable name or path.

    Examples
    --------
    >>> client = FFprobeClient()                          # doctest: +SKIP
    >>> client.probe_duration(Path("clip.mp4"))           # doctest: +SKIP
    12.5
    """

    def __init__(
        self,
        subprocess_module: types.ModuleType = subprocess,
        *,
        timeout: int = 30,
        ffprobe_bin: str = "ffprobe",
    ) -> None:
        self._subprocess = subprocess_module
        self._timeout = timeout
        self._ffprobe_bin = ffprobe_bin

    def probe(
        self,
        path: str | Path,
        *,
        verbose_level: str = "quiet",
        show_streams: bool = False,
        show_format: bool = True,
        entries: str | None = None,
        timeout: int | None = None,
    ) -> dict:
        """Run ffprobe and return the parsed JSON.

        Parameters
        ----------
        path:
            Media file to probe.
        verbose_level:
            Value for ``-v`` (``"quiet"`` or ``"error"``).
        show_streams:
            Include ``-show_streams``.
        show_format:
            Include ``-show_format``.
        entries:
            When set, passed as ``-show_entries <entries>`` instead of the
            ``show_format`` / ``show_streams`` pair.
        timeout:
            Per-call timeout override in seconds.

        Returns
        -------
        dict
            ffprobe's JSON output.

        Raises
        ------
        FFprobeError
            On a non-zero exit code or unparseable JSON.
        subprocess.TimeoutExpired
            When every attempt exceeded *timeout*.
        """
        cmd: list[str] = [self._ffprobe_bin, "-v", verbose_level]

        if entries is not None:
            cmd += ["-show_entries", entries, "-of", "json"]
        else:
            cmd.append("-print_format")
            cmd.append("json")
            if show_format:
                cmd.append("-show_format")
            if show_streams:
                cmd.append("-show_streams")

        cmd.append(str(path))

        result = self._run_with_retry(cmd, timeout or self._timeout)

        if result.returncode != 0:
            raise FFprobeError(
                f"ffprobe failed (rc={result.returncode}): {(result.stderr or '')[:200]}"
            )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FFprobeError(f"ffprobe returned unparseable JSON: {exc}") from exc

    def _run_with_retry(self, cmd: list[str], timeout: int):
        """Run ffprobe, retrying ONLY on ``subprocess.TimeoutExpired``.

        A timeout reflects transient machine load, not corruption, so it is
        retried up to :data:`FFPROBE_TIMEOUT_RETRIES` times with a linear
        backoff. Any other exception, and any non-zero return code surfaced by
        the caller, is genuine and propagates immediately.

        When every attempt times out, ``TimeoutExpired`` is re-raised so a
        genuinely hung probe still surfaces — but the message says it was a
        timeout after retries rather than a damaged file, because the two get
        confused constantly.
        """
        last_exc: subprocess.TimeoutExpired | None = None
        for attempt in range(1, FFPROBE_TIMEOUT_RETRIES + 1):
            try:
                return self._subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                last_exc = exc
                if attempt < FFPROBE_TIMEOUT_RETRIES:
                    time.sleep(FFPROBE_TIMEOUT_BACKOFF_SEC * attempt)
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output=(
                f"ffprobe timed out after {FFPROBE_TIMEOUT_RETRIES} attempts "
                f"({timeout}s each) — transient machine load, not corruption. "
                f"Raise the client timeout if probes routinely exceed the limit "
                f"while an encode is running."
            ),
        ) from last_exc

    # Convenience helpers ---------------------------------------------------

    def probe_duration(self, path: str | Path) -> float:
        """Return the container duration in seconds, or ``0.0`` on any failure."""
        try:
            data = self.probe(path, show_format=True)
        except Exception:
            return 0.0
        try:
            return float(data.get("format", {}).get("duration", 0))
        except (TypeError, ValueError):
            return 0.0

    def probe_or_none(
        self,
        path: str | Path,
        *,
        show_streams: bool = False,
        show_format: bool = True,
    ) -> dict | None:
        """Like :meth:`probe`, but returns ``None`` instead of raising."""
        try:
            return self.probe(
                path,
                show_streams=show_streams,
                show_format=show_format,
            )
        except Exception:
            return None
