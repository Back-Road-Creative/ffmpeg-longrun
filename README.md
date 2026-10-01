# ffmpeg-longrun

**ffmpeg that fails loudly.** A small Python layer over `subprocess` for FFmpeg
jobs measured in hours rather than seconds — stall detection instead of timeout
guessing, progress telemetry, partial-output cleanup, moov-atom validation,
NVENC probing with a CPU fallback, and an audio path that can only turn things
down.

No Python dependencies. The `ffmpeg` and `ffprobe` binaries do the work.

---

## The problem

Wrapping a long encode in `subprocess.run(cmd, timeout=N)` makes you guess `N`,
and both guesses are wrong:

- **Guess low** and you kill healthy encodes. A six-hour 4K job on a machine
  that is also doing something else will blow through any timeout you were
  comfortable writing down.
- **Guess high** and a genuinely hung ffmpeg — a stalled hardware encoder, a
  network mount that went away, a filter deadlocked on a full pipe — holds the
  job for the whole ceiling before anyone notices.

The other half of the problem is that ffmpeg exiting `0` does not mean the file
is good. An encode killed at the wrong moment, or one that ran out of disk,
leaves an MP4 with no moov atom. It is the right size. It has the right name. It
fails to open, three stages later, after you deleted the source.

`ffmpeg-longrun` measures **progress** rather than elapsed time, and checks the
**artifact** rather than the exit code.

---

## Install

```bash
pip install ffmpeg-longrun
```

Requires Python 3.11+, and `ffmpeg` / `ffprobe` on `PATH`:

```bash
# Debian / Ubuntu
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
```

NVENC hardware encoding additionally needs an NVIDIA GPU, a supported driver,
and an ffmpeg built with `--enable-nvenc`. Without one, the library falls back
to x265 on the CPU — see [Hardware encoding](#hardware-encoding).

---

## Usage

### Run a long encode and check the result

```python
from pathlib import Path

from ffmpeg_longrun import (
    build_encode_args,
    nvenc_available,
    run_ffmpeg_encode,
    validate_video,
)

source = Path("source.mov")
output = Path("delivery.mp4")
duration = 4 * 60 * 60  # four hours

cmd = [
    "ffmpeg", "-y", "-i", str(source),
    *build_encode_args(height=2160, has_nvenc=nvenc_available()),
    str(output),
]

result = run_ffmpeg_encode(
    cmd,
    output_path=output,
    expected_duration=duration,
    description="delivery master",
    stall_timeout=600,      # 10 minutes with no progress = dead
    max_timeout=86_400,     # absolute backstop
)

if not result.success:
    # output has already been deleted — there is no half-file to trip over
    raise SystemExit(f"encode failed ({result.killed_reason}): {result.stderr_tail}")

check = validate_video(output, expected_duration=duration)
if not check.valid:
    raise SystemExit(f"encode produced an unusable file: {check.reason}")

print(f"{result.wall_time / 3600:.1f}h at {result.last_speed:.2f}x realtime")
```

### Watch progress

```python
def on_progress(current_s, total_s, speed, fps):
    pct = current_s / total_s * 100 if total_s else 0
    print(f"\r{pct:5.1f}%  {fps:.0f} fps  {speed:.2f}x", end="")

run_ffmpeg_encode(cmd, output, duration, progress_callback=on_progress)
```

An exception raised inside your callback is swallowed. A broken progress meter
must not kill a six-hour encode.

### Probe a file

```python
from ffmpeg_longrun import FFprobeClient

probe = FFprobeClient()
print(probe.probe_duration("clip.mp4"))
print(probe.probe("clip.mp4", show_streams=True)["streams"])
```

`FFprobeClient` retries a probe that **times out** — on a loaded machine, a
perfectly good file can exceed a 30-second timeout while an encode pins every
core — and never retries one that **fails**, because a non-zero exit or
unparseable JSON is the file actually being broken.

### Attenuate audio without ever amplifying it

```python
from ffmpeg_longrun import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
    run_ffmpeg_quick,
)

measure = run_ffmpeg_quick(build_loudnorm_measure_cmd("source.mov"), timeout=900)
stats = parse_loudnorm_stats(measure.stderr)

af = build_attenuate_filter(stats)   # e.g. "volume=-31.73dB"
cmd = ["ffmpeg", "-y", "-i", "source.mov", "-af", af, "-c:v", "copy", "out.mp4"]
```

See [Attenuate-only audio](#attenuate-only-audio) for why this exists as its own
function rather than as a flag on a normaliser.

---

## What it does

### Stall detection

`run_ffmpeg_encode` reads ffmpeg's stderr on a background thread and resets a
timer on every progress line. An encode that keeps reporting frames survives for
as long as it likes; one that stops reporting dies within `stall_timeout`.
`max_timeout` remains as an absolute backstop for a process that reports
progress forever and will never finish.

One special case is handled: `-movflags +faststart` relocates the moov atom in a
second pass over the finished file, which reports **no** frame progress at all.
On a large 4K file that pass can run for many minutes. Left alone, the stall
detector kills a perfectly healthy job right at the finish line. The runner
detects the phase from ffmpeg's own log line, widens the window while it lasts,
and restores it the moment frame progress resumes.

### Partial-output cleanup

Every failure path — non-zero exit, stall kill, max-timeout kill, unhandled
exception, a command that could not be launched at all — deletes the output file
before returning. A half-written file that survives a failure is worse than no
file: it has plausible size and a valid name, and something downstream will
eventually treat it as finished.

### stdout draining

stdout is drained on its own thread whether or not you asked for it. ffmpeg
normally writes media to its output argument and says nothing on stdout, but
`-progress pipe:1`, `loudnorm print_format=json`, and `ffprobe -of json` all
emit there. With `stdout=PIPE` and nothing reading it, the OS pipe buffer (about
64 KB on Linux) fills and blocks the child mid-write. The symptom is a silent
hang that the stall detector eventually kills with no useful diagnostic — the
encode looks stuck, and the actual cause is a full pipe. Draining costs one
thread; whatever was captured lands on `FFmpegResult.stdout`.

Captured stdout is bounded. By default `FFmpegResult.stdout` keeps the newest
`max_stdout_bytes` (1 MiB) of what the child wrote; if more was written,
`stdout_truncated` is `True` and `stdout_bytes` is the total the child produced.
That is enough for an `ffprobe -of json` blob or a loudnorm report. For output
you cannot afford to lose, or a stream that runs for the life of a multi-hour
encode (`-progress pipe:1`), pass a `stdout_sink` — a text file or a callable
taking one `str` — and every chunk is delivered to it instead of being buffered:

```python
with open("progress.log", "w") as log:
    result = run_ffmpeg_encode(
        cmd + ["-progress", "pipe:1"], output, duration, stdout_sink=log
    )
assert result.stdout == "" and not result.stdout_truncated
```

A sink that raises does not hang the child (the drain keeps reading), but the
output after the failure is discarded and `stdout_sink_error` says so. stdout is
decoded as UTF-8 with replacement: it is for text. Write media to the output
file, never to stdout.

### Validation

`validate_video` runs ffprobe and checks, in order: the file exists and is
non-empty; ffprobe can parse it at all (**this is the moov-atom check**); a
video stream is present; an audio stream is present if you required one; the
per-stream audio and video durations agree; the container duration matches what
you asked for.

The A/V drift check is the one that is easy to skip and expensive to omit. A
container-level duration check passes a file whose streams have drifted apart —
the overall duration still looks right, and the symptom is audio running ahead
of picture, which nothing downstream detects.

`validate_video` never raises: not for an unusable file, and not for a missing
ffprobe. It returns a `VideoValidation` whose `reason` explains the verdict.

### Hardware encoding

`nvenc_available()` is a **real probe** — it asks ffmpeg to encode one frame of
a synthetic source with the hardware encoder and reports whether that worked.
Detecting an NVIDIA card, or grepping `ffmpeg -encoders`, both lie: the encoder
can be compiled in and listed while the driver refuses to open a session, which
happens with a driver/runtime version mismatch, with every NVENC session already
in use, or in a container without the device mapped through.

The probe uses a 256x256 test source rather than something smaller because NVENC
rejects frames below a driver-dependent minimum, and a 1x1 probe would fail for
a reason unrelated to availability.

```python
from ffmpeg_longrun import build_encode_args, nvenc_available, require_nvenc

args = build_encode_args(height=2160, has_nvenc=nvenc_available())
# → hevc_nvenc p7 CQ 18 main10, or libx265 slow CRF 18 main10 on the CPU

require_nvenc()   # raises NVENCUnavailableError instead of falling back
```

**The fallback is real but it is not free.** x265 `slow` on a multi-hour 4K
source can take most of a day where NVENC takes a few hours, and lands a few
percent behind on quality at the same nominal setting. `build_encode_args` logs
a warning when it falls back, because the failure mode is a job that "is still
running" and nobody knowing why. `require_nvenc()` refuses to start instead.

### Attenuate-only audio

The obvious way to hit a loudness target is two-pass `loudnorm`: measure, then
normalise to `I=<target>`. That filter moves loudness in **both** directions —
it will happily amplify quiet audio up to the target. For most delivery work
that is exactly right.

It is the wrong tool when quiet is the requirement rather than the starting
point. Consider published footage from a dashcam, a cockpit, a helmet, or a
fixed site camera: the interesting content is visual, and the microphone has
picked up passengers, bystanders, and background music that nobody consented to
publishing. The goal is ambience without intelligible speech.

Point `loudnorm` at that and it does its job faithfully in the wrong direction:
barely-audible conversation is normalised **up** to the target and becomes
perfectly clear in the delivered file. The encode succeeds, the loudness
measurement passes, and the defect surfaces only when someone plays the result
with the volume up.

`build_attenuate_filter` removes the possibility. It computes one uniform gain,
clamps it at zero, and raises if the clamp is ever violated:

```python
build_attenuate_filter({"input_i": -30.0, "input_tp": -12.0})  # 'volume=-15.00dB'
build_attenuate_filter({"input_i": -60.0, "input_tp": -40.0})  # 'volume=0.00dB'  (already quiet)
build_attenuate_filter(None)                                   # 'volume=-25.00dB' (measurement failed)
```

There is no configuration that makes it amplify. That is why it is a separate
function and not a flag on a normaliser.

**Checking the result.** Integrated loudness is a weak gate on intelligibility:
a handful of loud words barely move a whole-file average, so a file can pass an
integrated check while individual phrases stay perfectly clear. Gate on the
loudest momentary window as well — run `ffmpeg -af ebur128` over the finished
file and compare the highest `M:` value against
`LoudnessTargets.momentary_ceiling_lufs`. A uniformly attenuated file sits far
below it; one that was never attenuated trips it even when its integrated
loudness looks fine.

The shipped defaults (-45 LUFS integrated, -10 dBTP true peak, -30 LUFS
momentary) suit that privacy case and sit far below any broadcast standard.
Raise them toward a normal delivery target if that is not your problem — the
attenuate-only guarantee holds at any setting.

---

## Configuration

Quality settings live in two frozen dataclasses. The shipped values are
defaults, not policy:

```python
from ffmpeg_longrun import DEFAULT_VIDEO_PROFILE, LoudnessTargets, build_encode_args

# 4K HEVC delivery: p7 / CQ 18 / main10 / 10-bit, 40 Mbps floor at 2160p
draft = DEFAULT_VIDEO_PROFILE.replace(nvenc_preset="p4", nvenc_cq=26, cpu_crf=26)
args = build_encode_args(2160, has_nvenc=True, profile=draft)

broadcast = LoudnessTargets(integrated_ceiling_lufs=-23.0, true_peak_ceiling_dbtp=-1.0)
```

`VERTICAL_SOCIAL_VIDEO_PROFILE` keeps the same quality settings with the hard
absolute bitrate caps short vertical clips need. Matching the master's quality
matters more than it looks: an 8-bit re-encode of graded footage bands visibly
in skies and gradients, and a platform's own re-encode makes that worse rather
than better.

Every field is documented on `VideoProfile` and `LoudnessTargets`. One warning
worth repeating here: **do not switch to CBR.** In CBR mode ffmpeg drops `-cq`,
and NVENC combined with `-tune hq` then produces a fraction of the intended
bitrate — observed on consumer RTX hardware at under 1 Mbps where roughly
90 Mbps was expected. The encode succeeds and the file looks fine until you
inspect it.

---

## API

| Name | What it does |
| --- | --- |
| `run_ffmpeg_encode(cmd, output_path, expected_duration, ...)` | Long encode with stall detection, progress parsing, cleanup. Returns `FFmpegResult`. |
| `run_ffmpeg_quick(cmd, output_path=None, timeout=300, ...)` | Short operation with a timeout. Returns `CompletedProcess`, raises `RuntimeError` on failure. |
| `check_disk_space(path, required_bytes)` | Preflight; raises `RuntimeError` when the filesystem is too full. |
| `validate_video(path, expected_duration=None, ...)` | ffprobe integrity check. Returns `VideoValidation`; never raises. |
| `FFprobeClient(...)` | Parameterised ffprobe wrapper. `.probe()`, `.probe_duration()`, `.probe_or_none()`. |
| `nvenc_available(...)` | Real one-frame hardware-encoder probe. Returns `bool`. |
| `require_nvenc(...)` | Same probe; raises `NVENCUnavailableError` instead of returning `False`. |
| `build_encode_args(height, has_nvenc=..., profile=..., audio=...)` | Encoder half of an ffmpeg command; NVENC or x265. |
| `build_nvenc_args(profile)` / `build_cpu_args(profile)` | The two codec argument sets on their own. |
| `build_loudnorm_measure_cmd(path, targets=...)` | Audio-only pass-1 `loudnorm` command. |
| `parse_loudnorm_stats(stderr)` | Pulls the JSON stats block out of ffmpeg's stderr. |
| `build_attenuate_filter(measured, targets=...)` | Non-positive `volume=` gain. Never amplifies. |
| `VideoProfile` / `LoudnessTargets` | Frozen settings dataclasses; `.replace(**changes)` to customise. |

---

## Limits

- **Linux and macOS are what this is tested on.** Nothing is deliberately
  POSIX-only, but the pipe-buffer behaviour that motivates the stdout drainer is
  described in Linux terms and the CI runs on Ubuntu.
- **Progress parsing is regex over ffmpeg's human-readable stderr.** That format
  is stable in practice but is not a documented interface. If you need
  guarantees, use `-progress pipe:1` and stream it with `stdout_sink` (or parse
  `FFmpegResult.stdout`, which keeps only the newest `max_stdout_bytes`).
- **Stall detection cannot tell a hung encoder from a very slow one.** If a
  legitimately slow filter chain goes longer than `stall_timeout` between
  progress lines, raise the timeout. The default of 10 minutes is generous for
  frame-based work and too tight for some analysis passes.
- **NVENC only.** There is no AMD (AMF), Intel (QSV), or VideoToolbox path. The
  hardware fallback is CPU x265, not another accelerator.
- **`validate_video` checks structure, not pixels.** It will pass a file that is
  the right length with the right streams and entirely green. Use it as a
  cheap gate before an expensive irreversible step, not as quality control.
- **Loudness measurement decodes the whole file.** Expect many times realtime,
  not instant, on a long source.
- **No concurrency control.** Run two encodes at once and they will contend; the
  timeout retry in `FFprobeClient` exists precisely because that contention is
  normal, but nothing here schedules or throttles for you.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug fixes want a failing test first.

## License

MIT — see [LICENSE](LICENSE).
