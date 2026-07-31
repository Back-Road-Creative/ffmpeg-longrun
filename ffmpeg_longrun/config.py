"""Tunable defaults for encoding and loudness.

Every number here is a *default*, not a policy. The values ship as frozen
dataclasses so a caller can pin one profile for a whole run (and log it, or
fold it into a cache fingerprint) instead of threading a dozen keyword
arguments through the call stack.

Two profiles are provided out of the box:

``DEFAULT_VIDEO_PROFILE``
    High-quality 4K HEVC delivery, tuned for uploads to a streaming platform
    that will re-encode the file on its side. The reasoning: once the platform
    re-encodes, extra source bitrate stops buying visible quality, so the
    profile targets the point just past that knee rather than the maximum the
    encoder can produce.

``VERTICAL_SOCIAL_VIDEO_PROFILE``
    Same visual quality settings (10-bit, main10, CQ 18) with the absolute
    bitrate caps that short vertical clips need. Matching the master's quality
    matters more than it looks: an 8-bit re-encode of graded footage bands
    visibly in skies and gradients, and the platform's own re-encode makes
    that worse, not better.

``DEFAULT_LOUDNESS_TARGETS``
    Ceilings for the attenuate-only loudness path. See
    :mod:`ffmpeg_longrun.loudness` for what "attenuate-only" means and why the
    defaults sit far below any broadcast loudness standard.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "VideoProfile",
    "LoudnessTargets",
    "DEFAULT_VIDEO_PROFILE",
    "VERTICAL_SOCIAL_VIDEO_PROFILE",
    "DEFAULT_LOUDNESS_TARGETS",
]


@dataclass(frozen=True)
class VideoProfile:
    """Encoder settings shared by the NVENC and CPU paths.

    Attributes
    ----------
    nvenc_codec, cpu_codec:
        Encoder names passed to ``-c:v``. The CPU encoder is the fallback used
        whenever NVENC is unavailable (no NVIDIA GPU, no driver, a build of
        ffmpeg without ``--enable-nvenc``).
    nvenc_preset:
        NVENC preset ``p1`` (fastest) through ``p7`` (slowest, best quality).
    nvenc_cq:
        NVENC constant-quality level in VBR mode. Lower is better quality.
        18 is roughly visually transparent for high-detail 4K material.
    cpu_preset, cpu_crf:
        x265 equivalents of the two settings above.
    nvenc_pix_fmt, cpu_pix_fmt:
        Pixel formats. The two encoders name the same 10-bit format
        differently: NVENC takes the ``p010le`` input surface, x265 takes
        ``yuv420p10le``.
    encoder_profile:
        ``-profile:v``. ``main10`` is the HEVC 10-bit profile.
    color_range, colorspace, color_primaries, color_trc:
        Colour metadata tags written into the output. These *tag* the stream;
        they do not convert it. Set them to match what the filter chain
        actually produced, or players will misinterpret the levels.
    output_fps:
        Frame rate as an ffmpeg rational string, e.g. ``"30000/1001"`` for
        NTSC 29.97. Pinning output fps keeps a library of clips consistent
        even when sources were shot at mixed rates.
    gop_size:
        Keyframe interval in frames. 60 at 29.97 fps is a 2-second GOP, which
        is what most streaming platforms want for segmenting.
    bitrate_floor_2160p_kbps, bitrate_floor_1080p_kbps, bitrate_floor_kbps:
        Absolute VBR floor hints by output height, *not* derived from the
        source bitrate. Constant-quality encoding already adapts to scene
        complexity; the floor only stops the encoder from starving simple
        content (a long straight road, a flat sky) of bits.
    maxrate_ratio, bufsize_ratio:
        ``-maxrate`` and ``-bufsize`` as multiples of the floor. A maxrate too
        close to the floor pegs the encoder at its ceiling on every frame, so
        detail-heavy scenes cannot spend the bits the CQ setting is asking
        for. 2x headroom is enough for that to stop happening.
    maxrate_kbps, bufsize_kbps:
        Absolute overrides. When set, they win over the ratios — useful for
        short clips where a platform enforces a hard cap.
    audio_codec, audio_bitrate:
        Used only when ``audio="encode"`` is requested from
        :func:`ffmpeg_longrun.encoders.build_encode_args`.
    faststart:
        Append ``-movflags +faststart`` so the moov atom is relocated to the
        front of the file. Costs a second pass over the finished file; makes
        the result streamable without a full download.

    Notes
    -----
    Do not switch these settings to CBR. In CBR mode ffmpeg drops ``-cq``, and
    NVENC combined with ``-tune hq`` then produces a fraction of the intended
    bitrate (observed on consumer RTX hardware: under 1 Mbps where ~90 Mbps
    was expected). The failure is silent — the encode succeeds and the file
    looks fine until you inspect it.
    """

    nvenc_codec: str = "hevc_nvenc"
    nvenc_preset: str = "p7"
    nvenc_cq: int = 18
    nvenc_pix_fmt: str = "p010le"
    nvenc_tune: str = "hq"
    nvenc_lookahead: int = 32
    nvenc_spatial_aq: bool = True
    nvenc_temporal_aq: bool = True
    nvenc_b_frames: int = 4
    nvenc_b_ref_mode: str = "2"
    nvenc_multipass: str = "2"

    cpu_codec: str = "libx265"
    cpu_preset: str = "slow"
    cpu_crf: int = 18
    cpu_pix_fmt: str = "yuv420p10le"

    encoder_profile: str = "main10"

    color_range: str = "tv"
    colorspace: str = "bt709"
    color_primaries: str = "bt709"
    color_trc: str = "bt709"

    output_fps: str = "30000/1001"
    gop_size: int = 60

    bitrate_floor_2160p_kbps: int = 40_000
    bitrate_floor_1080p_kbps: int = 15_000
    bitrate_floor_kbps: int = 5_000
    maxrate_ratio: float = 2.0
    bufsize_ratio: float = 3.0
    maxrate_kbps: int | None = None
    bufsize_kbps: int | None = None

    audio_codec: str = "aac"
    audio_bitrate: str = "384k"

    faststart: bool = True

    def bitrate_floor_for_height(self, height: int) -> int:
        """Return the VBR floor in kbps for an output of *height* pixels."""
        if height >= 2160:
            return self.bitrate_floor_2160p_kbps
        if height >= 1080:
            return self.bitrate_floor_1080p_kbps
        return self.bitrate_floor_kbps

    def rate_control_kbps(self, height: int) -> tuple[int, int, int]:
        """Return ``(target, maxrate, bufsize)`` in kbps for *height*.

        Absolute overrides on the profile win over the ratio-derived values.
        """
        target = self.bitrate_floor_for_height(height)
        maxrate = (
            self.maxrate_kbps if self.maxrate_kbps is not None else int(target * self.maxrate_ratio)
        )
        bufsize = (
            self.bufsize_kbps if self.bufsize_kbps is not None else int(target * self.bufsize_ratio)
        )
        return target, maxrate, bufsize

    def replace(self, **changes: object) -> VideoProfile:
        """Return a copy with *changes* applied (thin wrap of ``dataclasses.replace``)."""
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LoudnessTargets:
    """Ceilings for the attenuate-only loudness path.

    The defaults are deliberately far below any broadcast or streaming
    loudness standard (which sit around -14 to -23 LUFS). They suit footage
    where *audible speech is a defect rather than a goal* — dashcam, cockpit,
    body-worn, wildlife, and site-survey recordings, where bystanders and
    passengers did not consent to being published and the ambience is the only
    part worth keeping.

    Raise every ceiling toward a normal delivery target if that is not your
    problem. The attenuate-only guarantee holds at any setting: the filter
    builder can move loudness down, never up.

    Attributes
    ----------
    integrated_ceiling_lufs:
        Integrated (whole-file) loudness the attenuation aims for. Material
        already quieter than this is left alone.
    true_peak_ceiling_dbtp:
        True-peak cap in dBTP. Tightens the gain further when a file's peaks
        are hot relative to its integrated loudness.
    momentary_ceiling_lufs:
        Ceiling on the loudest 400 ms window, as reported by ffmpeg's
        ``ebur128`` momentary (M) field. This exists because integrated
        loudness alone is a weak gate for intelligibility: a handful of loud
        words barely move a whole-file average, so a file can pass an
        integrated check while individual phrases stay perfectly clear. A
        uniformly attenuated file lands far below this; an unattenuated or
        under-attenuated one trips it.
    measurement_lra:
        Loudness range passed to the ``loudnorm`` *measurement* command. It
        does not affect the attenuation — ``loudnorm`` needs an ``LRA=``
        argument to accept the filter, and pass 1 only reports statistics.
    fallback_attenuation_db:
        Fixed reduction applied when measurement failed and no source loudness
        is known. Deliberately conservative: it must land typical material
        under the ceiling without knowing where it started.
    filter_algorithm_version:
        Bump this whenever the *shape* of the audio filter changes, not just
        its numbers. Callers that cache or resume long jobs can fold it into a
        fingerprint so a run started under the old rule re-renders instead of
        silently reusing output built by it.
    """

    integrated_ceiling_lufs: float = -45.0
    true_peak_ceiling_dbtp: float = -10.0
    momentary_ceiling_lufs: float = -30.0
    measurement_lra: float = 11.0
    fallback_attenuation_db: float = -25.0
    filter_algorithm_version: int = 1

    def __post_init__(self) -> None:
        if self.fallback_attenuation_db > 0:
            raise ValueError(
                "fallback_attenuation_db must be <= 0 — this path attenuates only, "
                f"got {self.fallback_attenuation_db:+.2f} dB"
            )

    def replace(self, **changes: object) -> LoudnessTargets:
        """Return a copy with *changes* applied (thin wrap of ``dataclasses.replace``)."""
        return replace(self, **changes)  # type: ignore[arg-type]


DEFAULT_VIDEO_PROFILE = VideoProfile()
"""4K HEVC delivery profile — see :class:`VideoProfile`."""

VERTICAL_SOCIAL_VIDEO_PROFILE = VideoProfile(
    maxrate_kbps=8_000,
    bufsize_kbps=12_000,
    audio_bitrate="128k",
)
"""Short vertical clips: same quality settings, hard absolute bitrate caps."""

DEFAULT_LOUDNESS_TARGETS = LoudnessTargets()
"""Near-silent ceilings for privacy-sensitive ambience — see :class:`LoudnessTargets`."""
