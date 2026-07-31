"""The attenuate-only loudness contract.

The single invariant worth protecting: ``build_attenuate_filter`` computes a
non-positive ``volume=`` gain and can never amplify. Material already below
the ceiling is left untouched (gain 0); louder material is pulled DOWN to the
integrated ceiling, tightened further by the true-peak cap when needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ffmpeg_longrun.config import DEFAULT_LOUDNESS_TARGETS, LoudnessTargets
from ffmpeg_longrun.loudness import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)

_VOLUME_RE = re.compile(r"^volume=(-?\d+(?:\.\d+)?)dB$")


def _gain_db(af: str) -> float:
    m = _VOLUME_RE.match(af)
    assert m, f"filter is not a bare volume= gain: {af!r}"
    return float(m.group(1))


# ── build_attenuate_filter: never amplifies ───────────────────────────


def test_never_returns_positive_gain_across_sweep():
    """A wide sweep of measured loudness / true peak never yields a boost."""
    for input_i in range(-70, -4, 5):
        for input_tp in range(-40, 1, 5):
            af = build_attenuate_filter({"input_i": input_i, "input_tp": input_tp})
            assert _gain_db(af) <= 0.0, (
                f"positive gain at input_i={input_i}, input_tp={input_tp}: {af!r}"
            )


def test_never_amplifies_at_a_relaxed_ceiling():
    """The guarantee holds at broadcast-style targets too, not only the default."""
    targets = LoudnessTargets(integrated_ceiling_lufs=-14.0, true_peak_ceiling_dbtp=-1.0)
    for input_i in range(-70, -4, 5):
        af = build_attenuate_filter({"input_i": input_i, "input_tp": -20.0}, targets=targets)
        assert _gain_db(af) <= 0.0, af


def test_quieter_than_ceiling_is_not_amplified():
    """Material already below the ceiling stays untouched (gain 0)."""
    af = build_attenuate_filter({"input_i": -50.0, "input_tp": -30.0})
    assert _gain_db(af) == 0.0, af


def test_louder_than_ceiling_is_attenuated_to_ceiling():
    """Gain == ceiling - input_i, unless the true-peak cap tightens it."""
    input_i = -20.0
    # input_tp chosen so the integrated term dominates.
    af = build_attenuate_filter({"input_i": input_i, "input_tp": -8.0})
    expected = DEFAULT_LOUDNESS_TARGETS.integrated_ceiling_lufs - input_i  # -45 - (-20) = -25
    assert _gain_db(af) == pytest.approx(expected), af


def test_true_peak_tightens_the_gain_further():
    """A hot true peak tightens the gain beyond the integrated reduction."""
    # input_i=-30 → integrated term = -45-(-30) = -15. input_tp=-2 → tp term =
    # -10-(-2) = -8 → the integrated term (-15) is MORE negative, so gain=-15.
    af = build_attenuate_filter({"input_i": -30.0, "input_tp": -2.0})
    assert _gain_db(af) == pytest.approx(-15.0), af


def test_true_peak_wins_when_it_is_the_binding_constraint():
    """When the peak needs more reduction than the integrated ceiling, it wins."""
    targets = LoudnessTargets(integrated_ceiling_lufs=-20.0, true_peak_ceiling_dbtp=-30.0)
    # integrated term = -20-(-25) = +5 → clamped to 0. tp term = -30-(-3) = -27.
    af = build_attenuate_filter({"input_i": -25.0, "input_tp": -3.0}, targets=targets)
    assert _gain_db(af) == pytest.approx(-27.0), af


def test_specific_quiet_source_value():
    af = build_attenuate_filter({"input_i": -30.0, "input_tp": -12.0})
    assert af == "volume=-15.00dB", af


def test_specific_loud_source_value():
    af = build_attenuate_filter({"input_i": -15.0, "input_tp": -2.0})
    assert af == "volume=-30.00dB", af


def test_already_below_ceiling_value():
    af = build_attenuate_filter({"input_i": -60.0, "input_tp": -40.0})
    assert af == "volume=0.00dB", af


def test_no_measured_uses_fixed_fallback():
    assert build_attenuate_filter(None) == "volume=-25.00dB"
    assert build_attenuate_filter({}) == "volume=-25.00dB"


def test_fallback_is_configurable():
    targets = LoudnessTargets(fallback_attenuation_db=-40.0)
    assert build_attenuate_filter(None, targets=targets) == "volume=-40.00dB"


def test_positive_fallback_is_rejected_at_construction():
    """A boosting fallback cannot be configured at all."""
    with pytest.raises(ValueError, match="attenuates only"):
        LoudnessTargets(fallback_attenuation_db=6.0)


def test_string_measurements_are_accepted():
    """ffmpeg reports its stats as JSON strings, not numbers."""
    af = build_attenuate_filter({"input_i": "-30.0", "input_tp": "-12.0"})
    assert af == "volume=-15.00dB", af


def test_unparseable_measurement_falls_back():
    """A garbage input_i is treated as no measurement, never as a boost."""
    af = build_attenuate_filter({"input_i": "-inf-ish", "input_tp": "-12.0"})
    assert af == "volume=-25.00dB", af


def test_missing_true_peak_uses_integrated_only():
    af = build_attenuate_filter({"input_i": -30.0})
    assert af == "volume=-15.00dB", af


def test_output_is_a_bare_volume_filter_with_no_loudnorm():
    """The filter must never contain loudnorm — that is the boosting path."""
    for input_i in (-8.0, -14.0, -20.0, -31.2, -50.0, -70.0):
        af = build_attenuate_filter({"input_i": input_i, "input_tp": -3.0})
        assert af.startswith("volume=")
        assert "loudnorm" not in af
        assert "linear=" not in af


# ── build_loudnorm_measure_cmd ────────────────────────────────────────


def test_measure_cmd_is_audio_only_and_prints_json():
    cmd = build_loudnorm_measure_cmd(Path("/clips/source.mp4"))
    assert cmd[0] == "ffmpeg"
    assert "-vn" in cmd, "measurement must not decode video"
    assert cmd[-2:] == ["-f", "null"] or cmd[-1] == "-"
    af = cmd[cmd.index("-af") + 1]
    assert af.startswith("loudnorm=")
    assert "print_format=json" in af
    assert "/clips/source.mp4" in cmd


def test_measure_cmd_carries_the_targets():
    targets = LoudnessTargets(
        integrated_ceiling_lufs=-23.0,
        true_peak_ceiling_dbtp=-2.0,
        measurement_lra=7.0,
    )
    af = build_loudnorm_measure_cmd("in.mp4", targets=targets)[
        build_loudnorm_measure_cmd("in.mp4", targets=targets).index("-af") + 1
    ]
    assert "I=-23.0" in af
    assert "TP=-2.0" in af
    assert "LRA=7.0" in af


def test_measure_cmd_honours_a_custom_ffmpeg_binary():
    cmd = build_loudnorm_measure_cmd("in.mp4", ffmpeg_bin="/opt/tools/ffmpeg")
    assert cmd[0] == "/opt/tools/ffmpeg"


# ── parse_loudnorm_stats ──────────────────────────────────────────────

_SAMPLE_STDERR = """\
[Parsed_loudnorm_0 @ 0x5583d0]
{
	"input_i" : "-13.27",
	"input_tp" : "-3.24",
	"input_lra" : "11.70",
	"input_thresh" : "-23.60",
	"output_i" : "-45.00",
	"target_offset" : "0.70"
}
"""


def test_parse_extracts_the_stats_block():
    stats = parse_loudnorm_stats(_SAMPLE_STDERR)
    assert stats is not None
    assert stats["input_i"] == "-13.27"
    assert stats["input_tp"] == "-3.24"
    assert stats["target_offset"] == "0.70"


def test_parse_returns_none_when_no_block_present():
    assert parse_loudnorm_stats("ffmpeg version 6.0\nnothing useful here\n") is None


def test_parse_raises_on_malformed_block():
    """A present-but-broken block is a different failure from an absent one."""
    broken = '[Parsed_loudnorm_0]\n{\n\t"input_i" : "-13.27",,, }\n'
    with pytest.raises(json.JSONDecodeError):
        parse_loudnorm_stats(broken)


def test_measure_then_attenuate_round_trip():
    """The two halves compose: parse ffmpeg's stderr, build the gain from it."""
    stats = parse_loudnorm_stats(_SAMPLE_STDERR)
    af = build_attenuate_filter(stats)
    # input_i -13.27 → integrated term -45-(-13.27) = -31.73
    # input_tp  -3.24 → tp term -10-(-3.24) = -6.76 → integrated term wins
    assert _gain_db(af) == pytest.approx(-31.73, abs=0.01)
