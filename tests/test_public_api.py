"""The package's advertised surface exists and is importable.

If a name is in ``__all__`` and the README, a consumer will import it. This
catches a rename that never made it into ``__init__.py``.
"""

import importlib

import ffmpeg_longrun


def test_version_is_the_released_one():
    assert ffmpeg_longrun.__version__ == "0.1.0"


def test_every_exported_name_resolves():
    for name in ffmpeg_longrun.__all__:
        assert hasattr(ffmpeg_longrun, name), f"__all__ advertises missing name: {name}"


def test_submodules_import_cleanly():
    for mod in ("config", "runner", "validate", "probe", "encoders", "loudness"):
        importlib.import_module(f"ffmpeg_longrun.{mod}")


def test_no_third_party_runtime_dependencies():
    """The package must import with nothing but the standard library present.

    Every top-level import in the package is checked against the stdlib module
    list, so a stray ``import numpy`` cannot slip into a release that claims
    zero runtime dependencies.
    """
    import ast
    import sys
    from pathlib import Path

    pkg_dir = Path(ffmpeg_longrun.__file__).parent
    stdlib = set(sys.stdlib_module_names)
    offenders = []

    for source in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in stdlib:
                        offenders.append(f"{source.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative, in-package
                    continue
                root = (node.module or "").split(".")[0]
                if root and root not in stdlib:
                    offenders.append(f"{source.name}: from {node.module} import ...")

    assert not offenders, f"non-stdlib imports found: {offenders}"


def test_doctests_in_the_public_modules_pass():
    """The README's promises are duplicated as doctests in the docstrings."""
    import doctest

    for name in ("ffmpeg_longrun.loudness", "ffmpeg_longrun.encoders"):
        module = importlib.import_module(name)
        results = doctest.testmod(module, verbose=False)
        assert results.failed == 0, f"{name}: {results.failed} doctest failure(s)"
