#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the RoboWits EFE suite's upstream assets.

RoboWits (UMass-Embodied-AGI/RoboWits, MIT) runs bimanual physical-reasoning
puzzles for the Tianji Marvin bimanual robot on Genesis 0.4.7. Its Apache-2.0
asset bundle (robot URDF + glb meshes, worktable textures) is published as the
HuggingFace dataset ``XHRlyb2001/RoboWits_assets`` (pinned
``b1b49abba3bda9f3b91982d37139d62b3de9a543``; fetched via hf-mirror.com, the
host's huggingface.co route is blocked). The per-family scene builders and the
``_check_success`` predicates live in the RoboWits repo (pinned
``9cc30aedbbebea86b975c0d66ea9ac227ca38535``); the EFE wrappers
(``rcb_robowits.py``) live in ``embodied_first_exam/benchmarks/robowits``.

Three of the 15 in-repo families (dominos / stand_pages / stack_cubes) build
from primitives + the bundle alone; the other 12 additionally need
non-redistributable BlenderKit models and stay pending.

This tool mirrors, idempotently:

* the HF asset bundle -> ``assets/robowits/`` (raw glb + textures; the
  glb->obj conversion for the MuJoCo probe lives in ``tools/rw_worlds.py``);
* the RoboWits sources -> ``data/efe/upstream/rw-src/robowits-git``
  (stage clone, gs_gym robowits task modules);
* the EFE tooling + task files -> ``data/efe/upstream/rw-src/efe-benchmark/``
  and ``data/efe/upstream/tasks/robowits/``.

Like the other fetch tools the mirrors stay local and uncommitted; CI and
tests never call this tool.

Usage::

    uv run python tools/fetch_rw_assets.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".efe_stage"
RW_GIT = STAGE / "robowits-git"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)
RW_HF_CACHE = STAGE / "rw_assets"

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
SRC_OUT = UPSTREAM_OUT / "rw-src"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "robowits"
ASSETS_OUT = ROOT / "assets" / "robowits"

RW_COMMIT = "9cc30aedbbebea86b975c0d66ea9ac227ca38535"
HF_REPO = "XHRlyb2001/RoboWits_assets"
HF_REV = "b1b49abba3bda9f3b91982d37139d62b3de9a543"
HF_MIRROR = "https://hf-mirror.com"

_RW_GIT_RELS = [
    "gs_gym/envs/robowits",
    "gs_gym/robots",
    "gs_gym/configs/envs/robowits_default.yaml",
    "gs_gym/robots/schema.py",
]
_EFE_BENCHMARK_RELS = [
    "privileged/tests/rcb_robowits.py",
    "sim/server.py",
]

_TASK_FILES = (
    "environment/instance.json",
    "environment/sim/instance.json",
    "instruction.md",
    "task.toml",
    "README.md",
)


def _log(msg: str) -> None:
    print(f"[fetch-rw] {msg}", flush=True)


def _check_sources() -> None:
    import subprocess

    if not RW_GIT.is_dir():
        raise SystemExit(f"stage clone missing: {RW_GIT}")
    head = subprocess.run(
        ["git", "-C", str(RW_GIT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head.startswith(RW_COMMIT):
        raise SystemExit(f"{RW_GIT} at {head!r}, expected {RW_COMMIT}")
    if not (RW_HF_CACHE / "marvin_bimanual/urdf/marvin_pika.urdf").is_file():
        raise SystemExit(
            "HF asset cache missing; fetch the bundle first "
            "(see tools/fetch_rw_assets.py::_mirror_hf_bundle)"
        )


def _mirror_files(pairs: list[tuple[Path, Path]]) -> int:
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


def _mirror_hf_bundle() -> None:
    """Mirror the pinned HF asset bundle (hf-mirror.com) into the stage cache."""
    if (RW_HF_CACHE / "marvin_bimanual/urdf/marvin_pika.urdf").is_file():
        _log(f"hf bundle: cached at {RW_HF_CACHE}")
        return
    RW_HF_CACHE.mkdir(parents=True, exist_ok=True)
    tree_url = f"{HF_MIRROR}/api/datasets/{HF_REPO}/tree/{HF_REV}?recursive=true"
    out = subprocess.run(
        [
            "curl",
            "-sL",
            "--noproxy",
            "*",
            "-H",
            "User-Agent: Mozilla/5.0",
            "-o",
            str(RW_HF_CACHE / "_tree.json"),
            tree_url,
        ],
        check=True,
    )
    del out
    import json

    tree = json.loads((RW_HF_CACHE / "_tree.json").read_text(encoding="utf-8"))
    files = [e["path"] for e in tree if e["type"] == "file"]
    base = f"{HF_MIRROR}/datasets/{HF_REPO}/resolve/{HF_REV}/"
    fails = []
    for rel in files:
        if rel == ".gitattributes":
            continue
        dst = RW_HF_CACHE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "curl",
                "-sL",
                "--noproxy",
                "*",
                "-H",
                "User-Agent: Mozilla/5.0",
                "-o",
                str(dst),
                base + rel.replace(" ", "%20"),
            ]
        )
        if r.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
            fails.append(rel)
    (RW_HF_CACHE / "_tree.json").unlink(missing_ok=True)
    _log(f"hf bundle: {len(files) - len(fails)}/{len(files)} files -> {RW_HF_CACHE}")
    if fails:
        _log(f"hf bundle: {len(fails)} failed (textures are visual-only): {fails}")


def mirror_bundle() -> None:
    """Mirror the asset bundle (glb + textures) into assets/robowits/."""
    n = _mirror_tree(
        RW_HF_CACHE,
        ["marvin_bimanual", "worktable_texture"],
        ASSETS_OUT,
    )
    _log(f"bundle: mirrored {n} files -> {ASSETS_OUT.relative_to(ROOT)}")


def mirror_sources() -> None:
    """Mirror the RoboWits sources and the EFE wrappers."""
    n = _mirror_tree(RW_GIT, _RW_GIT_RELS, SRC_OUT / "robowits-git")
    for cand in EFE_STAGE_CANDIDATES:
        bench = cand / "embodied_first_exam" / "benchmarks" / "robowits"
        if bench.is_dir():
            break
    else:
        raise SystemExit("EFE benchmark checkout not found")
    n += _mirror_tree(bench, _EFE_BENCHMARK_RELS, SRC_OUT / "efe-benchmark")
    _log(f"sources: mirrored {n} files -> {SRC_OUT.relative_to(ROOT)}")


def mirror_efe_tasks() -> None:
    """Mirror the EFE task files for all 30 RoboWits task dirs."""
    for cand in EFE_STAGE_CANDIDATES:
        tasks = cand / "tasks" / "robowits"
        if tasks.is_dir():
            break
    else:
        raise SystemExit("EFE stage checkout not found; sparse-checkout tasks/robowits")
    n = 0
    for task_dir in sorted(tasks.iterdir()):
        if not task_dir.is_dir():
            continue
        for rel in _TASK_FILES:
            f = task_dir / rel
            if f.is_file():
                n += _mirror_files([(f, TASKS_OUT / task_dir.name / rel)])
    _log(f"tasks: mirrored {n} files -> {TASKS_OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _mirror_hf_bundle()
    _check_sources()
    mirror_bundle()
    mirror_sources()
    mirror_efe_tasks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
