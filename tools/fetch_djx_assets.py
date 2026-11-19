#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the DexJoCo EFE suite's upstream assets.

DexJoCo (brave-eai/dexjoco @ 8d23b0f, MIT) runs a Franka Panda arm with a
16-DoF Allegro hand (one or two arms) in ``arena_arm_hand_*.xml`` scenes.
Its EFE tasks freeze one demonstration's reset state into
``environment/instance.npz`` (the full ``mjSTATE_INTEGRATION`` vector,
mocap targets, ctrl/act, the env's Python-side attrs, and every numeric
``MjModel`` array the restorers touch), so the scene spec is the package's
env XML plus those array overlays — no simulator package install is needed
for the world pack.

Mirrors, idempotently:

* the upstream package -> ``data/efe/upstream/djx-src/dexjoco``
  (``sim/envs`` with the xmls trees, ``sim/controllers``,
  ``sim/mujoco_gym_env.py``, LICENSE — byte-identical at the pinned commit);
* EFE task files (``instance.npz``, ``instruction.md``, ``task.toml``,
  ``README.md``, ``solution/``) -> ``data/efe/upstream/tasks/dexjoco/``;
* the spawn-side xmls tree -> ``assets/dexjoco/xmls`` (the composed world
  XMLs' mesh references point here).

The stage clone lives at ``.efe_stage/brave-eai_dexjoco-git``.

Usage::

    uv run python tools/fetch_djx_assets.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".efe_stage"
CLONE = STAGE / "brave-eai_dexjoco-git"
PKG = CLONE / "dexjoco" / "dexjoco"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
SRC_OUT = UPSTREAM_OUT / "djx-src" / "dexjoco"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "dexjoco"
ASSETS_OUT = ROOT / "assets" / "dexjoco"

DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"

_TASK_FILES = (
    "environment/instance.npz",
    "instruction.md",
    "task.toml",
    "README.md",
    "solution/gen.py",
    "solution/solve.sh",
)


def _log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-djx] {msg}", flush=True)


def _check_commit() -> None:
    if not PKG.is_dir():
        raise SystemExit(f"stage clone missing: {CLONE}")
    head = subprocess.run(
        ["git", "-C", str(CLONE), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != DEXJOCO_COMMIT:
        raise SystemExit(f"dexjoco clone at {head!r}, expected {DEXJOCO_COMMIT}")


def _mirror_files(pairs: list[tuple[Path, Path]]) -> int:
    """Copy (src, dst) pairs that are new or changed; return the copy count."""
    n = 0
    for src, dst in pairs:
        if not src.is_file():
            raise SystemExit(f"mirror source missing: {src}")
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def mirror_sources() -> None:
    """Mirror the dexjoco package subset the port reads."""
    n = 0
    for rel in (
        "sim/envs",
        "sim/controllers",
        "sim/mujoco_gym_env.py",
        "sim/rendering.py",
        "sim/__init__.py",
        "__init__.py",
    ):
        src = PKG / rel
        dst = SRC_OUT / Path(rel)
        if src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    n += _mirror_files([(f, dst / f.relative_to(src))])
        elif src.is_file():
            n += _mirror_files([(src, dst)])
        else:
            raise SystemExit(f"mirror source missing: {src}")
    n += _mirror_files([(CLONE / "LICENSE", SRC_OUT / "repo_LICENSE")])
    n += _mirror_files(
        [(CLONE / "dexjoco" / "pyproject.toml", SRC_OUT / "pyproject.toml")]
    )
    _log(f"sources: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}")


def mirror_efe_tasks() -> None:
    """Mirror the EFE task files for all 22 DexJoCo task dirs."""
    for cand in EFE_STAGE_CANDIDATES:
        tasks = cand / "tasks" / "dexjoco"
        if tasks.is_dir():
            break
    else:
        raise SystemExit("EFE stage checkout not found; sparse-checkout tasks/dexjoco")
    n = 0
    for task_dir in sorted(tasks.iterdir()):
        if not task_dir.is_dir():
            continue
        for rel in _TASK_FILES:
            f = task_dir / rel
            if f.is_file():
                n += _mirror_files([(f, TASKS_OUT / task_dir.name / rel)])
        oracle = task_dir / "solution" / "oracle_trajectory.npz"
        if oracle.is_file():
            n += _mirror_files(
                [
                    (
                        oracle,
                        TASKS_OUT
                        / task_dir.name
                        / "solution"
                        / "oracle_trajectory.npz",
                    )
                ]
            )
    _log(f"tasks: mirrored {n} files -> {TASKS_OUT.relative_to(ROOT)}")


def mirror_spawn_assets() -> None:
    """Mirror the env xmls tree the composed world XMLs reference."""
    src = PKG / "sim" / "envs" / "xmls"
    dst = ASSETS_OUT / "xmls"
    n = 0
    for f in sorted(src.rglob("*")):
        if f.is_file():
            n += _mirror_files([(f, dst / f.relative_to(src))])
    _log(f"assets: mirrored {n} files -> {dst.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _check_commit()
    mirror_sources()
    mirror_efe_tasks()
    mirror_spawn_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
