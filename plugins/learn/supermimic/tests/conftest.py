"""Test bootstrap for the supermimic plugin: expose the ``supermimic`` package.

The plugin keeps its sources under ``core/`` (top-level package name), so
tests add ``core/`` to ``sys.path`` — same convention as the maniskill
plugin tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))
