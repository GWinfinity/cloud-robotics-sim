#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the ABC Sim EFE suite's upstream assets.

ABC Sim is the MuJoCo benchmark of ``amazon-far/abc`` (the simulator package
``abc_sim`` @ d0832d1, Apache-2.0; assets CC BY 4.0 / MIT). Its EFE tasks are
frozen ``instance.npz`` files (assembled scene XML + qpos0 + randomization
record), so the simulator package itself is reference material (success
evaluators in ``abc_sim/task_eval/``; runtimes in ``abc_sim/task_runtime/``)
and the meshes the scene XMLs open come from the checksum-pinned asset
packages of the EFE image recipe.

This tool mirrors, idempotently:

* EFE task files (``instance.npz``, ``task.json``, ``instruction.md``,
  ``task.toml``, ``README.md``, ``solution/``) for all 60 task dirs ->
  ``data/efe/upstream/tasks/abc-sim/``;
* the image build recipe (``Dockerfile``, ``VERSION``, ``assemble.py``,
  ``assets.txt``, ``ASSET_LICENSES.md``, ``tools/``) ->
  ``data/efe/upstream/images-abc-sim/`` (the archive SHA-256 table is parsed
  from the mirrored Dockerfile, the single source of truth);
* the upstream source tree @ d0832d1 (``abc_sim/`` + ``LICENSE``,
  byte-identical) -> ``data/efe/upstream/abc-src/``;
* the 22 asset archives, SHA-256 verified, with only the files
  ``assets.txt`` lists (plus licence/attribution files) unpacked ->
  ``assets/abc-sim/models/``.

Like the other fetch tools the mirrors stay local and uncommitted; only this
tool is. CI and tests never call it. Network is flaky by design: per-request
retries with backoff, resume partial downloads, skip files already present
and matching their SHA-256.

Usage::

    uv run python tools/fetch_abc_assets.py            # everything
    uv run python tools/fetch_abc_assets.py --tasks-only
    uv run python tools/fetch_abc_assets.py --assets-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "abc-sim"
IMAGES_OUT = UPSTREAM_OUT / "images-abc-sim"
SRC_OUT = UPSTREAM_OUT / "abc-src"
ASSETS_OUT = ROOT / "assets" / "abc-sim" / "models"

STAGE = ROOT / ".efe_stage"
EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)
ABC_CLONE = STAGE / "amazon-far_abc-git"

ABC_COMMIT = "d0832d12651d1b260a652861a14648dc5f3660c7"

_TASK_FILES = (
    "environment/instance.npz",
    "environment/sim/instance.npz",
    "environment/sim/task.json",
    "instruction.md",
    "task.toml",
    "README.md",
    "solution/oracle.npy",
    "solution/oracle_stream.py",
    "solution/oracle_replay.py",
    "solution/solve.sh",
)

_RECIPE_FILES = (
    "Dockerfile",
    "VERSION",
    "assemble.py",
    "assets.txt",
    "ASSET_LICENSES.md",
)

_LICENCE = re.compile(r"(^|/)(LICENSE[^/]*|README\.md|[^/]*ATTRIBUTIONS[^/]*\.md)$")

_ARCHIVE_RE = re.compile(
    r"ADD --checksum=sha256:([0-9a-f]{64})\s+(https://[^\s]+/([^\s/]+\.tar\.gz))\s+/pkgs/"
)

DEADLINE = time.time() + 3 * 3600


def log(msg: str) -> None:
    print(f"[fetch-abc] {msg}", flush=True)


def stage_dir() -> Path:
    for c in EFE_STAGE_CANDIDATES:
        if (c / "tasks" / "abc-sim").is_dir():
            return c
    raise SystemExit("EFE stage checkout not found; sparse-checkout tasks/abc-sim first")


def mirror_efe_tasks() -> None:
    src = stage_dir() / "tasks" / "abc-sim"
    n = 0
    for task_dir in sorted(src.iterdir()):
        if not task_dir.is_dir():
            continue
        for rel in _TASK_FILES:
            f = task_dir / rel
            if not f.is_file():
                continue
            dst = TASKS_OUT / task_dir.name / rel
            if dst.exists() and dst.stat().st_size == f.stat().st_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    log(f"tasks: mirrored {n} files -> {TASKS_OUT.relative_to(ROOT)}")


def mirror_recipe() -> None:
    src = stage_dir() / "images" / "abc-sim"
    n = 0
    for rel in _RECIPE_FILES:
        f = src / rel
        if not f.is_file():
            raise SystemExit(f"image recipe file missing: {f}")
        dst = IMAGES_OUT / rel
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    tools_dst = IMAGES_OUT / "tools"
    for f in sorted((src / "tools").glob("*.py")):
        dst = tools_dst / f.name
        if dst.exists() and dst.stat().st_size == f.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    log(f"recipe: mirrored {n} files -> {IMAGES_OUT.relative_to(ROOT)}")


def mirror_source() -> None:
    if not ABC_CLONE.is_dir():
        raise SystemExit(f"upstream clone not found at {ABC_CLONE}")
    head = (
        shutil.which("git")
        and __import__("subprocess")
        .run(
            ["git", "-C", str(ABC_CLONE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    if head != ABC_COMMIT:
        raise SystemExit(f"abc clone at {head!r}, expected {ABC_COMMIT}")
    n = 0
    for rel in ("abc_sim", "LICENSE"):
        src = ABC_CLONE / rel
        for f in sorted(src.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            dst = SRC_OUT / rel / f.relative_to(src)
            if dst.exists() and dst.stat().st_size == f.stat().st_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    log(f"source: mirrored {n} files @ {ABC_COMMIT[:7]} -> {SRC_OUT.relative_to(ROOT)}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_table() -> dict[str, tuple[str, str]]:
    """{name: (url, sha256)} parsed from the mirrored image recipe."""
    text = (IMAGES_OUT / "Dockerfile").read_text(encoding="utf-8")
    table = {m.group(3): (m.group(2), m.group(1)) for m in _ARCHIVE_RE.finditer(text)}
    if not table:
        raise SystemExit("no pinned archives parsed from images-abc-sim/Dockerfile")
    return table


def download(url: str, dest: Path, sha256: str) -> None:
    if dest.exists() and sha256_of(dest) == sha256:
        log(f"  have {dest.name} ({dest.stat().st_size} B)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(8):
        if time.time() > DEADLINE:
            raise SystemExit("download deadline exceeded")
        try:
            req = urllib.request.Request(url)
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            if tmp.exists():
                req.add_header("Range", f"bytes={tmp.stat().st_size}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if tmp.exists() and resp.status == 206 else "wb"
                if mode == "wb":
                    tmp.unlink(missing_ok=True)
                with tmp.open(mode) as f:
                    shutil.copyfileobj(resp, f, 1 << 20)
            if sha256_of(tmp) == sha256:
                tmp.replace(dest)
                log(f"  got {dest.name} ({dest.stat().st_size} B)")
                return
            log(f"  checksum mismatch on {dest.name}, retrying ({attempt + 1})")
            tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - network flakiness by design
            wait = min(2**attempt, 60)
            log(f"  {dest.name}: {type(exc).__name__}: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"failed to download {url}")


def mirror_assets() -> None:
    table = archive_table()
    keep = {
        line.strip()
        for line in (IMAGES_OUT / "assets.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    pkgs = UPSTREAM_OUT / "abc-pkgs"
    pkgs.mkdir(parents=True, exist_ok=True)
    log(f"archives: {len(table)} pinned packages")
    for name, (url, sha256) in sorted(table.items()):
        download(url, pkgs / name, sha256)

    found: set[str] = set()
    n = 0
    for tgz in sorted(pkgs.glob("*.tar.gz")):
        with tarfile.open(tgz) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                name = m.name.lstrip("./")
                rel = name if name.startswith("assets/") else f"assets/{name}"
                if rel not in keep and not _LICENCE.search(rel):
                    continue
                dst = ASSETS_OUT / rel
                if dst.exists() and dst.stat().st_size == m.size:
                    found.add(rel)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as fsrc, dst.open("wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, 1 << 20)
                found.add(rel)
                n += 1
    missing = sorted(keep - found)
    if missing:
        raise SystemExit(f"asset files not found in the packages: {missing[:10]} ({len(missing)})")
    log(f"assets: extracted {n} new files ({len(found)} total) -> {ASSETS_OUT.relative_to(ROOT)}")

    # The probe venv runs upstream abc_sim, whose env anchors asset lookup at
    # the package's own ``abc_sim/models`` dir (env.py setup_model chdir);
    # link the mirror there (hardlinks: no extra disk) so the original scene
    # XML strings compile exactly like in the EFE image. ``ASSETS_OUT`` is
    # ``models/assets`` (mirror layout matches the image's), so files link
    # directly under ``models/``.
    link_root = SRC_OUT / "abc_sim" / "models"
    link_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for src in sorted(ASSETS_OUT.rglob("*")):
        if not src.is_file():
            continue
        dst = link_root / src.relative_to(ASSETS_OUT)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        linked += 1
    if linked:
        log(f"source tree: linked {linked} asset files -> {link_root.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-only", action="store_true")
    ap.add_argument("--assets-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.assets_only:
        mirror_efe_tasks()
        mirror_recipe()
        mirror_source()
    if not args.tasks_only:
        mirror_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
