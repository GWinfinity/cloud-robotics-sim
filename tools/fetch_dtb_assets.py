#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the DexToolBench EFE suite's upstream assets.

DexToolBench (SimToolReal, MIT) runs a KUKA iiwa 14 arm with a 22-DoF Sharpa
HA4 hand on a table, carrying a tool through an ordered goal trajectory. Its
EFE tasks carry only ``environment/instance.json`` (start pose + ordered
goals); the scene is rebuilt by SimToolReal's sim2sim recipe
(``deployment/mujoco/mujoco_sim.py``) from:

* ``tylerlum/simtoolreal @ 313d5ae`` — the Sharpa HA4 hand (offset variant),
  the DexToolBench object URDFs (offline-decomposed collision), the per-task
  table environment URDFs, the goal trajectories, and the scene recipe;
* ``google-deepmind/mujoco_menagerie @ 1b86ece`` — the KUKA iiwa 14 MJCF the
  hand attaches to (``attachment_site``).

Both clones live under ``.efe_stage/`` (fetched once, byte-identical). This
tool mirrors, idempotently:

* the upstream sources -> ``data/efe/upstream/dtb-src/{simtoolreal,menagerie}``;
* EFE task files (``instance.json``, ``instruction.md``, ``task.toml``,
  ``README.md``) for all 42 task dirs -> ``data/efe/upstream/tasks/dextoolbench/``;
* the spawn-side asset tree the composed world XMLs reference ->
  ``assets/dextoolbench/``.

Like the other fetch tools the mirrors stay local and uncommitted; CI and
tests never call this tool.

Usage::

    uv run python tools/fetch_dtb_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".efe_stage"
SIMTOOLREAL = STAGE / "tylerlum_simtoolreal-git"
MENAGERIE = STAGE / "mujoco_menagerie-git"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
SRC_OUT = UPSTREAM_OUT / "dtb-src"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "dextoolbench"
ASSETS_OUT = ROOT / "assets" / "dextoolbench"

SIMTOOLREAL_COMMIT = "313d5ae"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"

_TASK_FILES = (
    "environment/instance.json",
    "environment/sim/instance.json",
    "instruction.md",
    "task.toml",
    "README.md",
)


def _log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-dtb] {msg}", flush=True)


def _check_commits() -> None:
    """Both stage clones must sit at the pinned commits."""
    import subprocess

    for repo, ref in ((SIMTOOLREAL, SIMTOOLREAL_COMMIT), (MENAGERIE, MENAGERIE_COMMIT)):
        if not repo.is_dir():
            raise SystemExit(f"stage clone missing: {repo}")
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not head.startswith(ref[: len(head)]):
            raise SystemExit(f"{repo} at {head!r}, expected {ref}")


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


def _mirror_tree(src_root: Path, rels: list[str], dst_root: Path) -> int:
    """Mirror selected subtree members (dirs are copied recursively)."""
    n = 0
    for rel in rels:
        src = src_root / rel
        dst = dst_root / rel
        if src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    n += _mirror_files([(f, dst / f.relative_to(src))])
        elif src.is_file():
            n += _mirror_files([(src, dst)])
        else:
            raise SystemExit(f"mirror source missing: {src}")
    return n


def mirror_sources() -> None:
    """Mirror the two upstream sources at their pinned commits."""
    n = _mirror_tree(
        SIMTOOLREAL,
        [
            "assets/urdf/left_sharpa_ha4",
            "assets/urdf/dextoolbench",
            "dextoolbench/objects.py",
            "dextoolbench/trajectories",
            "deployment/mujoco/mujoco_sim.py",
            "LICENSE",
        ],
        SRC_OUT / "simtoolreal",
    )
    n += _mirror_tree(
        MENAGERIE,
        ["kuka_iiwa_14", "LICENSE"],
        SRC_OUT / "menagerie",
    )
    _log(f"sources: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}")


def mirror_efe_tasks() -> None:
    """Mirror the EFE task files for all 42 DexToolBench task dirs."""
    for cand in EFE_STAGE_CANDIDATES:
        tasks = cand / "tasks" / "dextoolbench"
        if tasks.is_dir():
            break
    else:
        raise SystemExit(
            "EFE stage checkout not found; sparse-checkout tasks/dextoolbench"
        )
    n = 0
    for task_dir in sorted(tasks.iterdir()):
        if not task_dir.is_dir():
            continue
        for rel in _TASK_FILES:
            f = task_dir / rel
            if f.is_file():
                n += _mirror_files([(f, TASKS_OUT / task_dir.name / rel)])
    _log(f"tasks: mirrored {n} files -> {TASKS_OUT.relative_to(ROOT)}")


def mirror_spawn_assets() -> None:
    """Mirror the spawn-side asset tree the world XMLs reference."""
    n = _mirror_tree(
        SRC_OUT / "simtoolreal",
        ["assets/urdf/left_sharpa_ha4", "assets/urdf/dextoolbench"],
        ASSETS_OUT / "simtoolreal",
    )
    n += _mirror_tree(
        SRC_OUT / "menagerie",
        ["kuka_iiwa_14"],
        ASSETS_OUT / "menagerie",
    )
    _log(f"assets: mirrored {n} files -> {ASSETS_OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _check_commits()
    mirror_sources()
    mirror_efe_tasks()
    mirror_spawn_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
