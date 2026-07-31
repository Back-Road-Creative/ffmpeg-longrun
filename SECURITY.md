# Security Policy

## Supported versions

Only the latest released tag receives security fixes. `main` is unstable and
unsupported for production use.

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do **not** open a public
issue for a security report.

Open a private advisory via GitHub's **Security → Report a vulnerability** tab
on this repository. That channel is private to the maintainers until an
advisory is published.

Please include the affected version or commit, a description of the issue and
its impact, reproduction steps, and any suggested remediation.

## What to expect

- Acknowledgement within 5 business days.
- An initial assessment and severity triage within 10 business days.
- Coordinated disclosure: we will agree a timeline with you before any public
  write-up, and credit reporters who want it.

## Threat model

This library builds and runs subprocess command lines. Two things follow from
that, and they are the most likely source of a real problem:

**Command construction is the caller's responsibility.** Every function takes an
argument *list*, never a shell string, and nothing here invokes a shell. That
removes shell metacharacter injection. It does **not** remove argument
injection: if you interpolate untrusted input into a command list, a value
beginning with `-` can be read by ffmpeg as an option rather than a filename.
Validate untrusted paths, or prefix them with `./`, before passing them in.

**ffmpeg parses untrusted media.** The decoders are the attack surface for
hostile input, not this wrapper. Keep ffmpeg current, and consider running
untrusted media in a sandbox with a restricted filesystem view. `-protocol_whitelist`
and `-f` are worth setting when you know what you expect.

Also worth knowing:

- `run_ffmpeg_encode` and `run_ffmpeg_quick` **delete the output path you give
  them** on failure. Pass a path you own. Do not pass a directory, a symlink to
  something you care about, or a file another process is writing.
- Failure messages include the tail of ffmpeg's stderr, which contains input
  file paths. Consider that before logging them somewhere public.

In scope: the code under `ffmpeg_longrun/` and the CI workflow.

Out of scope: vulnerabilities in ffmpeg, ffprobe, or NVIDIA drivers (report
those upstream), and the security of media files this library reports on.
