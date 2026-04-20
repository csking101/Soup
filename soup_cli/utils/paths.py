"""Cross-platform path containment helpers.

Uses ``os.path.realpath`` + ``os.path.commonpath`` rather than
``Path.resolve() + relative_to()`` to survive Windows 8.3 short names
(e.g. ``C:\\Users\\RUNNER~1``) that can appear in one of the two paths
but not the other on older Python.

Why this lives in one place: the same helper is needed by autopilot,
registry, cans, eval-gate, quant-check, and trace harvesting. Keeping it
in a single module guarantees a single behaviour across the CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def is_under(path: Union[str, Path], base: Union[str, Path]) -> bool:
    """Return True when ``path`` resolves inside ``base``."""
    try:
        resolved_path = os.path.realpath(str(path))
        resolved_base = os.path.realpath(str(base))
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        resolved_path = resolved_path.lower()
        resolved_base = resolved_base.lower()
    try:
        return os.path.commonpath([resolved_path, resolved_base]) == resolved_base
    except ValueError:
        return False


def is_under_cwd(path: Union[str, Path]) -> bool:
    """Whether ``path`` is inside the current working directory."""
    return is_under(path, Path.cwd())
