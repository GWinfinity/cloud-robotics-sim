#!/usr/bin/env python3
"""One-time setup for the LIBERO-MAX compile-time probe venv (``.lmxvenv``).

The probe venv runs LIBERO-PRO + robosuite 1.4.0 on Windows purely as a
*compile-time* oracle: it rebuilds a frozen case's MuJoCo scene, applies the
``.pruned_init`` state, and dumps entity poses / joint values / goal-predicate
results for ``tools/compile_efe_suite.py`` (the committed Genesis configs never
import robosuite or LIBERO). Two wheels need a local, uncommitted touch that
this script reproduces idempotently:

* robosuite 1.4.0 expects ``robosuite/utils/mujoco.dll`` on Windows; the
  pinned ``mujoco==3.2.6`` wheel ships it, so we copy it over;
* robosuite's default ``MUJOCO_GPU_RENDERING=True`` forces ``MUJOCO_GL=egl``
  on Windows, which its own GL validator rejects — we flip the macro to
  ``False`` (physics-only probing never renders).

Usage::

    uv venv --python 3.10 .lmxvenv
    uv pip install --python .lmxvenv/Scripts/python.exe -r .efe_stage/mll-lab-nu_embodied-first-exam-git/images/libero-max/requirements.txt
    uv run python tools/setup_lmx_probe_venv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

VENV = Path(__file__).resolve().parents[1] / ".lmxvenv"


def main() -> int:
    """Apply the two wheel-local fixes; idempotent."""
    site = VENV / "Lib" / "site-packages"
    dll_src = site / "mujoco" / "mujoco.dll"
    dll_dst = site / "robosuite" / "utils" / "mujoco.dll"
    if not dll_src.is_file():
        print(f"missing {dll_src} — install mujoco==3.2.6 into .lmxvenv first")
        return 1
    if not dll_dst.is_file():
        dll_dst.write_bytes(dll_src.read_bytes())
        print(f"copied {dll_src.name} -> {dll_dst}")
    macros = site / "robosuite" / "macros.py"
    text = macros.read_text(encoding="utf-8")
    if "MUJOCO_GPU_RENDERING = True" in text:
        macros.write_text(
            text.replace("MUJOCO_GPU_RENDERING = True", "MUJOCO_GPU_RENDERING = False"),
            encoding="utf-8",
        )
        print("set MUJOCO_GPU_RENDERING = False in robosuite/macros.py")
    print("probe venv ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
