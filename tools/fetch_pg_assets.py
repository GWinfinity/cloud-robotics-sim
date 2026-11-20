#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the MuJoCo Playground EFE suite's upstream assets.

MuJoCo Playground (google-deepmind, Apache-2.0) runs quadruped locomotion
(Go1 / Spot / Barkour vB from MuJoCo Menagerie) and ALOHA / Franka Panda
manipulation tasks. Its EFE port loads Playground's scene XMLs the way
Playground's ``get_assets()`` does (bare file names from ``xmls/`` and the
Menagerie robot directory) with the drive gains Playground's environments
set; the graders and services live in the EFE repo's
``embodied_first_exam/benchmarks/mujoco-playground`` (``rcb_quadruped.py`` /
``rcb_aloha.py`` / ``rcb_panda.py``). The Playground scenes are pinned at
``4057c14`` and Menagerie at ``1b86ece`` (the commit Playground pins), the
same pins as the EFE image's ``images/mujoco/prepare_assets.sh``.

Both clones live under ``.efe_stage/`` (fetched once, byte-identical). This
tool mirrors, idempotently:

* the upstream sources -> ``data/efe/upstream/pg-src/{playground,menagerie}``
  (only the subtrees the world packs reference);
* the EFE benchmark tooling (``rcb_*.py``, graders) ->
  ``data/efe/upstream/pg-src/efe-benchmark/``;
* EFE task files (``instance.json``, ``instruction.md``, ``task.toml``,
  ``README.md``) for all 27 task dirs ->
  ``data/efe/upstream/tasks/mujoco-playground/``;
* the spawn-side asset tree the composed world XMLs reference ->
  ``assets/playground/``.

Like the other fetch tools the mirrors stay local and uncommitted; CI and
tests never call this tool.

Usage::

    uv run python tools/fetch_pg_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".efe_stage"
PLAYGROUND = STAGE / "mujoco_playground-git"
MENAGERIE = STAGE / "mujoco_menagerie-git"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
SRC_OUT = UPSTREAM_OUT / "pg-src"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "mujoco-playground"
ASSETS_OUT = ROOT / "assets" / "playground"

PLAYGROUND_COMMIT = "4057c14"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"

#: Playground subtrees the world packs load (mirrors prepare_assets.sh).
PLAYGROUND_RELS = [
    "mujoco_playground/_src/locomotion/go1/xmls",
    "mujoco_playground/_src/locomotion/spot/xmls",
    "mujoco_playground/_src/manipulation/aloha/xmls",
    "mujoco_playground/_src/manipulation/franka_emika_panda/xmls",
    "mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/xmls",
    "mujoco_playground/_src/gait.py",
    "LICENSE",
]
#: Menagerie robot dirs the Playground scenes compose (prepare_assets.sh).
MENAGERIE_RELS = [
    "unitree_go1",
    "boston_dynamics_spot",
    "google_barkour_vb",
    "aloha",
    "franka_emika_panda",
    "LICENSE",
]
#: EFE benchmark tooling that owns the task specs (graders + services).
EFE_BENCHMARK_RELS = [
    "privileged/environment/rcb_quadruped.py",
    "privileged/environment/rcb_aloha.py",
    "privileged/environment/rcb_panda.py",
    "privileged/tests/grade.py",
    "sim/server.py",
    "sim/server_mj.py",
    "sim/video.py",
    "sim/finalize.sh",
]

_TASK_FILES = (
    "environment/instance.json",
    "environment/sim/instance.json",
    "instruction.md",
    "task.toml",
    "README.md",
)


def _log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-pg] {msg}", flush=True)


def _check_commits() -> None:
    """Both stage clones must sit at the pinned commits."""
    import subprocess

    for repo, ref in ((PLAYGROUND, PLAYGROUND_COMMIT), (MENAGERIE, MENAGERIE_COMMIT)):
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
    """Mirror the pinned upstream subtrees."""
    n = _mirror_tree(PLAYGROUND, PLAYGROUND_RELS, SRC_OUT / "playground")
    n += _mirror_tree(MENAGERIE, MENAGERIE_RELS, SRC_OUT / "menagerie")
    _log(f"sources: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}")


def mirror_efe_benchmark() -> None:
    """Mirror the EFE benchmark tooling that owns the task specs."""
    for cand in EFE_STAGE_CANDIDATES:
        bench = cand / "embodied_first_exam" / "benchmarks" / "mujoco-playground"
        if bench.is_dir():
            break
    else:
        raise SystemExit(
            "EFE benchmark checkout not found; sparse-checkout "
            "embodied_first_exam/benchmarks/mujoco-playground"
        )
    n = _mirror_tree(bench, EFE_BENCHMARK_RELS, SRC_OUT / "efe-benchmark")
    _log(
        f"efe tooling: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}/efe-benchmark"
    )


def mirror_efe_tasks() -> None:
    """Mirror the EFE task files for all 27 MuJoCo Playground task dirs."""
    for cand in EFE_STAGE_CANDIDATES:
        tasks = cand / "tasks" / "mujoco-playground"
        if tasks.is_dir():
            break
    else:
        raise SystemExit(
            "EFE stage checkout not found; sparse-checkout tasks/mujoco-playground"
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
        SRC_OUT / "playground",
        ["mujoco_playground/_src"],
        ASSETS_OUT / "playground",
    )
    n += _mirror_tree(
        SRC_OUT / "menagerie",
        [
            "unitree_go1",
            "boston_dynamics_spot",
            "google_barkour_vb",
            "aloha",
            "franka_emika_panda",
        ],
        ASSETS_OUT / "menagerie",
    )
    _log(f"assets: mirrored {n} files -> {ASSETS_OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _check_commits()
    mirror_sources()
    mirror_efe_benchmark()
    mirror_efe_tasks()
    mirror_spawn_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
