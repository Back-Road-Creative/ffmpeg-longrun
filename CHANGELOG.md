# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- `run_ffmpeg_encode` no longer buffers stdout without bound. `FFmpegResult.stdout`
  keeps the newest `max_stdout_bytes` (default 1 MiB); `stdout_bytes` and
  `stdout_truncated` report what the child wrote and whether any was dropped.
  New `stdout_sink` streams stdout to a file or callable instead of retaining it.
  A failing sink is reported on `stdout_sink_error` and cannot deadlock the
  child; undecodable bytes no longer stop the stdout drain.

## 0.1.0

First public release.

### Added

- `run_ffmpeg_encode` — long-encode runner built on `Popen` with threaded pipe
  readers. Detects a stall from the gap since the last progress line rather
  than from total elapsed time, so an encode that keeps reporting survives
  indefinitely and one that stops dies in minutes. `max_timeout` remains as an
  absolute backstop. Widens the stall window automatically during the
  `+faststart` moov-atom rewrite, which reports no frame progress at all.
- Partial-output cleanup on every failure path: non-zero exit, stall kill,
  max-timeout kill, unhandled exception, and a command that could not be
  launched.
- Unconditional stdout draining, so a child writing more than the OS pipe
  buffer cannot deadlock. Captured output is returned on `FFmpegResult.stdout`.
- `run_ffmpeg_quick` — timeout-bounded wrapper for short operations, with the
  same cleanup guarantee.
- `check_disk_space` — preflight that walks up to the nearest existing
  ancestor, so it works before the output directory exists.
- `validate_video` / `VideoValidation` — ffprobe integrity check covering
  existence, non-zero size, parseability (the moov-atom case), stream presence,
  per-stream A/V duration drift, and container duration against an expectation.
  Returns a verdict rather than raising, including when ffprobe is not
  installed.
- `FFprobeClient` / `FFprobeError` — one parameterised ffprobe wrapper. Retries
  a timeout (transient machine load), never retries a failure (genuine
  corruption).
- `nvenc_available` / `require_nvenc` / `NVENCUnavailableError` — real one-frame
  hardware-encoder probe, with the choice of falling back to CPU or refusing to
  start.
- `build_encode_args` / `build_nvenc_args` / `build_cpu_args` — encoder argument
  builders for NVENC HEVC and x265, with resolution-based VBR floors, colour
  metadata tagging, and `copy` / `encode` / `none` audio modes.
- `build_loudnorm_measure_cmd` / `parse_loudnorm_stats` — two-pass `loudnorm`
  measurement primitives.
- `build_attenuate_filter` — computes a `volume=` gain that is guaranteed
  non-positive, for footage where audible bystander speech is a privacy defect
  rather than an audio-quality goal. Raises rather than ever emitting a boost.
- `VideoProfile` / `LoudnessTargets` — frozen configuration dataclasses with
  documented defaults, plus `DEFAULT_VIDEO_PROFILE`,
  `VERTICAL_SOCIAL_VIDEO_PROFILE`, and `DEFAULT_LOUDNESS_TARGETS`.
