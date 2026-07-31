"""Loudness measurement, and an audio filter that can only turn things down.

Three pieces:

:func:`build_loudnorm_measure_cmd`
    Builds the audio-only pass-1 ``loudnorm`` command. ffmpeg prints the
    file's integrated loudness, true peak, loudness range, and threshold to
    stderr as JSON, and writes no output.

:func:`parse_loudnorm_stats`
    Pulls that JSON block back out of the stderr text.

:func:`build_attenuate_filter`
    Turns a measurement into a single ``volume=NdB`` gain that is guaranteed
    non-positive.

Why attenuate-only
------------------
The obvious way to hit a loudness target is two-pass ``loudnorm``: measure,
then normalise to ``I=<target>``. That filter moves loudness in *both*
directions — it will happily amplify quiet audio up to the target. For most
delivery work that is exactly what you want.

It is the wrong tool when quiet is the requirement rather than the starting
point. Consider published footage from a dashcam, a cockpit, a helmet, or a
fixed site camera: the interesting content is visual, and the microphone has
picked up passengers, bystanders, and background music that nobody consented
to publishing. The goal is ambience without intelligible speech.

Point ``loudnorm`` at that and it does its job faithfully in the wrong
direction: barely-audible conversation gets normalised *up* to the target and
becomes perfectly clear in the delivered file. The encode succeeds, the
loudness measurement passes, and the defect only shows up when someone plays
the result with the volume up.

:func:`build_attenuate_filter` removes the possibility. It computes one gain,
clamps it at zero, and raises if the clamp is ever violated. Material already
below the ceiling is left alone. There is no configuration that makes it
amplify — that is the point of it existing as a separate function rather than
as a flag on a normaliser.

Checking the result
-------------------
Integrated loudness is a weak gate on intelligibility, because a few loud
words barely move a whole-file average. Gate on the loudest momentary window
as well: run ``ffmpeg -af ebur128`` over the finished file and compare the
highest ``M:`` value against
:attr:`~ffmpeg_longrun.config.LoudnessTargets.momentary_ceiling_lufs`. A
uniformly attenuated file sits far below it; one that was never attenuated
trips it even when its integrated loudness looks fine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import DEFAULT_LOUDNESS_TARGETS, LoudnessTargets

__all__ = [
    "build_loudnorm_measure_cmd",
    "parse_loudnorm_stats",
    "build_attenuate_filter",
]

# The pass-1 stats block as ffmpeg writes it to stderr: a JSON object whose
# first key is "input_i". Matched non-greedily up to the first closing brace.
_LOUDNORM_STATS_RE = re.compile(r'\{\s*\n\s*"input_i".*?\}', re.DOTALL)


def build_loudnorm_measure_cmd(
    input_path: str | Path,
    *,
    targets: LoudnessTargets = DEFAULT_LOUDNESS_TARGETS,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build the audio-only ``loudnorm`` measurement command for *input_path*.

    Decodes audio only (``-vn``), runs ``loudnorm=...:print_format=json`` so
    ffmpeg prints the pass-1 statistics to stderr, and discards output to
    null. The ``I``/``TP``/``LRA`` values in the filter are required syntax
    for ``loudnorm`` to accept it; in pass 1 they do not affect the reported
    measurement.

    The whole file is decoded, so this is not free on a long source — expect
    it to run at many times realtime but not instantly.
    """
    af = (
        f"loudnorm=I={targets.integrated_ceiling_lufs}"
        f":TP={targets.true_peak_ceiling_dbtp}"
        f":LRA={targets.measurement_lra}"
        f":print_format=json"
    )
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        af,
        "-f",
        "null",
        "-",
    ]


def parse_loudnorm_stats(stderr: str) -> dict | None:
    """Extract the pass-1 ``loudnorm`` statistics from ffmpeg's *stderr*.

    Returns the parsed dict — ``input_i``, ``input_tp``, ``input_lra``,
    ``input_thresh``, and ``target_offset`` — or ``None`` when no statistics
    block is present at all (the caller should warn and fall back to the fixed
    attenuation).

    Raises ``json.JSONDecodeError`` when a block is present but malformed, so
    a caller can tell "ffmpeg printed nothing" apart from "ffmpeg printed
    something broken".
    """
    json_match = _LOUDNORM_STATS_RE.search(stderr)
    if not json_match:
        return None
    return json.loads(json_match.group())


def _safe_float(measured: dict | None, key: str) -> float | None:
    if not measured or key not in measured:
        return None
    try:
        return float(measured[key])
    except (TypeError, ValueError):
        return None


def build_attenuate_filter(
    measured: dict | None,
    *,
    targets: LoudnessTargets = DEFAULT_LOUDNESS_TARGETS,
) -> str:
    """Build an ``-af`` filter string that can never amplify audio.

    Computes a single non-positive ``volume=`` gain that pulls the measured
    integrated loudness down to
    :attr:`~ffmpeg_longrun.config.LoudnessTargets.integrated_ceiling_lufs`,
    tightening further when the true peak would otherwise land above
    :attr:`~ffmpeg_longrun.config.LoudnessTargets.true_peak_ceiling_dbtp`.
    Material already quieter than the ceiling is returned untouched
    (``volume=0.00dB``).

    Because the gain is uniform, every momentary window moves down by the same
    number of decibels as the integrated loudness — so attenuating to the
    integrated ceiling also drags brief speech peaks down with it.

    Parameters
    ----------
    measured:
        The dict from :func:`parse_loudnorm_stats` (``input_i``, ``input_tp``).
        When ``None`` or empty — measurement failed — a fixed conservative
        attenuation is applied instead of any boost.
    targets:
        Ceilings to attenuate toward.

    Returns
    -------
    str
        An ffmpeg filter string such as ``"volume=-15.00dB"``.

    Raises
    ------
    RuntimeError
        If the computed gain is positive. This cannot happen given the clamps
        above; the check is there so that a future edit which breaks the
        guarantee fails at filter-build time rather than in a shipped file.

    Examples
    --------
    >>> build_attenuate_filter({"input_i": -30.0, "input_tp": -12.0})
    'volume=-15.00dB'
    >>> build_attenuate_filter({"input_i": -60.0, "input_tp": -40.0})
    'volume=0.00dB'
    >>> build_attenuate_filter(None)
    'volume=-25.00dB'
    """
    ceiling = targets.integrated_ceiling_lufs
    tp_cap = targets.true_peak_ceiling_dbtp
    input_i = _safe_float(measured, "input_i")
    input_tp = _safe_float(measured, "input_tp")

    if input_i is None:
        gain_db = targets.fallback_attenuation_db
    else:
        gain_db = min(0.0, ceiling - input_i)
        if input_tp is not None:
            # After a uniform volume gain, output true peak is roughly
            # input_tp + gain. Tighten if the integrated reduction alone would
            # leave the peak above the cap.
            gain_db = min(gain_db, tp_cap - input_tp)
        gain_db = min(0.0, gain_db)

    if gain_db > 0:
        raise RuntimeError(
            f"attenuate-only audio filter must never amplify (computed gain {gain_db:+.2f} dB)"
        )
    return f"volume={gain_db:.2f}dB"
