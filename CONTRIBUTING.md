# Contributing to ffmpeg-longrun

Thanks for your interest. This is a small library with a narrow job, so the
guidance is short.

## Reporting

- **Bugs and feature requests:** open an issue. For a bug, include your Python
  version, your `ffmpeg -version` output, and the exact command you passed in.
- **Security vulnerabilities:** do **not** open a public issue — follow
  [SECURITY.md](SECURITY.md).

## Development setup

From a clone of the repository, at its root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

`ffmpeg` and `ffprobe` on `PATH` are optional for development. Most of the
suite mocks the subprocess layer; the handful of tests that shell out to the
real binaries skip themselves when the binaries are absent.

## The gates

CI runs these on every push and pull request. Run them first:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

## What a good change looks like

- **Tests first for a bug fix.** Add a test that fails before your change and
  passes after. Never weaken or delete a test to make the suite pass.
- **Keep the dependency list empty.** This package imports nothing outside the
  standard library, and a test enforces that. If you genuinely need a
  dependency, open an issue first.
- **Do not make failures quiet.** The whole point of the library is that a
  broken encode is loud. A change that swallows an error, or returns a
  plausible-looking result where the old code raised, needs a very good reason
  in the pull request body.
- **Update the docs in the same change.** A new argument gets a docstring entry
  and, if a consumer would care, a README line. Docs and code do not land
  separately.
- **Add a CHANGELOG entry** under an `## Unreleased` heading.

## Style

`ruff` with the config in `pyproject.toml` (100-column lines, `py311` target).
Public functions carry NumPy-style docstrings — the docstring is where the
*why* goes, since the *what* is usually obvious from the signature.
