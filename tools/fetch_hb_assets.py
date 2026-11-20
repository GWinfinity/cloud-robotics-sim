#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the HumanoidBench EFE suite's upstream assets.

HumanoidBench (carlosferrazza/humanoid-bench, MIT, RSS 2024) runs whole-body
Unitree H1 / H1+Shadow-hand tasks in MuJoCo + dm_control. The EFE image
(``images/humanoidbench/``) pins upstream at ``cb118903`` and layers physics
fixes on top: ``no-torch.patch``, ``upstream-fixes.patch`` (room's reset,
powerlift's dumbbell mass) and ``box_feet.py`` (box feet on the
box-standing families). Rewards are scrubbed from the package but preserved
verbatim in ``rewards.json`` (verifier-side); the frozen instances live in
each task's ``environment/instance.npz``.

The stage clone (``.efe_stage/humanoid-bench-git``) sits at the pinned commit
with all EFE patches applied; this tool mirrors, idempotently:

* the patched upstream package -> ``data/efe/upstream/hb-src/humanoid-bench``
  (the importable tree the probe venv installs);
* the EFE image tooling -> ``data/efe/upstream/hb-src/efe-image/``
  (rcb_hb.py, rewards.json, patches, box_feet.py);
* EFE task files (``instance.npz``, ``instruction.md``, ``task.toml``,
  ``README.md``, ``tests/rewards.json``) for all 30 task dirs ->
  ``data/efe/upstream/tasks/humanoidbench/``;
* the spawn-side asset tree -> ``assets/humanoidbench/``.

Like the other fetch tools the mirrors stay local and uncommitted; CI and
tests never call this tool.

Usage::

    uv run python tools/fetch_hb_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".efe_stage"
HB = STAGE / "humanoid-bench-git"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
SRC_OUT = UPSTREAM_OUT / "hb-src"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "humanoidbench"
ASSETS_OUT = ROOT / "assets" / "humanoidbench"

HB_COMMIT = "cb1189039151c8aadaaa987b442da54383c87fab"

#: The importable package tree the probe venv installs (--no-deps, like the
#: EFE image); tests/dreamerv3/jaxrl_m/ppo training harnesses stay out.
HB_RELS = [
    "humanoid_bench",
    "setup.py",
    "README.md",
    "LICENSE",
]
#: EFE image tooling that owns the task specs / physics fixes.
EFE_IMAGE_RELS = [
    "rcb_hb.py",
    "rewards.json",
    "no-torch.patch",
    "upstream-fixes.patch",
    "scrub_oracles.py",
    "box_feet.py",
    "check_fixes.py",
    "VERSION",
]

_TASK_FILES = (
    "environment/instance.npz",
    "environment/sim/instance.npz",
    "instruction.md",
    "task.toml",
    "README.md",
    "tests/rewards.json",
)


def _log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-hb] {msg}", flush=True)


def _check_commit() -> None:
    """The stage clone must sit at the pinned commit with patches applied."""
    import subprocess

    if not HB.is_dir():
        raise SystemExit(f"stage clone missing: {HB}")
    head = subprocess.run(
        ["git", "-C", str(HB), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head.startswith(HB_COMMIT):
        raise SystemExit(f"{HB} at {head!r}, expected {HB_COMMIT}")
    marker = HB / "humanoid_bench" / "assets" / "envs" / "h1hand_pos_balance_hard.xml"
    if not marker.is_file():
        raise SystemExit("box_feet patch not applied to the stage clone")


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
    """Mirror the patched upstream package and the EFE image tooling."""
    n = _mirror_tree(HB, HB_RELS, SRC_OUT / "humanoid-bench")
    for cand in EFE_STAGE_CANDIDATES:
        image = cand / "images" / "humanoidbench"
        if image.is_dir():
            break
    else:
        raise SystemExit(
            "EFE image checkout not found; sparse-checkout images/humanoidbench"
        )
    n += _mirror_tree(image, EFE_IMAGE_RELS, SRC_OUT / "efe-image")
    _log(f"sources: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}")
    _swap_dmc_sizes()


def _swap_dmc_sizes() -> None:
    """Swap the vendored dm_control 1.0.20 sizes table for the installed
    dm_control's own (the EFE image does the same; the vendored table does
    not match mujoco 3.3.7's struct fields, e.g. ``flex_vert0``).
    """
    import dm_control

    src = (
        Path(dm_control.__file__).resolve().parent
        / "mujoco"
        / "wrapper"
        / "mjbindings"
        / "sizes.py"
    )
    dst = SRC_OUT / "humanoid-bench" / "humanoid_bench" / "dmc_deps" / "dmc_sizes.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    cache = dst.parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)
    _log("dmc_sizes: vendored table swapped for the installed dm_control's")


def mirror_efe_tasks() -> None:
    """Mirror the EFE task files for all 30 HumanoidBench task dirs."""
    for cand in EFE_STAGE_CANDIDATES:
        tasks = cand / "tasks" / "humanoidbench"
        if tasks.is_dir():
            break
    else:
        raise SystemExit(
            "EFE stage checkout not found; sparse-checkout tasks/humanoidbench"
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
        SRC_OUT / "humanoid-bench" / "humanoid_bench" / "assets",
        ["."],
        ASSETS_OUT / "humanoid_bench" / "assets",
    )
    _log(f"assets: mirrored {n} files -> {ASSETS_OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _check_commit()
    mirror_sources()
    mirror_efe_tasks()
    mirror_spawn_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
