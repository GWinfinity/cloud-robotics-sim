#!/usr/bin/env python3
"""Repo-local, idempotent mirror of the RoboCasa EFE suites' upstream assets.

Three EFE suites share this one tool: ``robocasa`` (RoboCasa v0.2),
``robocasa365`` (robocasa 1.0.1) and ``robocasa-gr1`` (robocasa-gr1-tabletop-tasks).
Each pins its own upstream stack (see ``tasks/<suite>/README.md`` in the EFE
repo and the mirrored image recipes):

* EFE task files (``instance.npz``, ``instruction.md``, ``solution/``) for the
  three suites -> ``data/efe/upstream/tasks/<suite>/``; image build recipes ->
  ``data/efe/upstream/images-robocasa/``;
* upstream sources -> ``data/efe/upstream/robocasa-src`` (robocasa @ 756598a,
  v0.2 stack), ``data/efe/upstream/robocasa365-src`` (robocasa @ 4f8a298,
  1.0.1 stack) and ``data/efe/upstream/robocasa-gr1-src``
  (robocasa-gr1-tabletop-tasks @ 4840e67);
* kitchen asset archives (fixtures / textures / object meshes), SHA-256
  pinned exactly as in the EFE image recipes, unpacked into each source
  tree's ``robocasa/models/assets`` (or ``models/assets`` for the GR1 repo)
  so a probe venv can ``pip install --no-deps -e`` the tree and build scenes.

Original UT Box download links are dead (404); where the image recipes used
them this tool substitutes RoboCasa's own Hugging Face mirror
(``robocasa/robocasa-assets``), which the EFE recipes themselves call
byte-identical for the overlapping archives. Archives whose only known
source is a dead link and for which no mirror is known are reported as
``MISSING`` and the suites' probe step decides whether they matter.

Like the other fetch tools the mirrors stay local and uncommitted; only this
tool is. CI and tests never call it. Network is flaky by design: per-request
retries with backoff, resume partial downloads, skip files already present
and matching their SHA-256.

Usage::

    uv run python tools/fetch_robocasa_assets.py           # tasks + sources + assets
    uv run python tools/fetch_robocasa_assets.py --tasks-only
    uv run python tools/fetch_robocasa_assets.py --assets-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = time.time() + 6 * 3600

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
IMAGES_OUT = UPSTREAM_OUT / "images-robocasa"
STAGE = ROOT / ".efe_stage"

EFE_STAGE_CANDIDATES = (
    STAGE / "mll-lab-nu_embodied-first-exam-git",
    STAGE / "efe-git",
)

SUITES = ("robocasa", "robocasa365", "robocasa-gr1")

# EFE task files worth mirroring (relative to each task dir).
_TASK_FILES = (
    "environment/instance.npz",
    "environment/sim/instance.npz",
    "instruction.md",
    "task.toml",
    "README.md",
    "solution/oracle.npz",
    "solution/solve.sh",
    "solution/oracle_replay.py",
)

_IMAGE_FILES = {
    "robocasa": ("Dockerfile", "fetch_objects.sh", "VERSION"),
    "robocasa365": ("Dockerfile", "fetch_assets.sh", "requirements.txt", "VERSION"),
    "robocasa-gr1": ("Dockerfile", "fetch_assets.sh", "requirements.txt", "VERSION"),
    "robocasa-files": ("FILES.json", "VERSION"),
    "robocasa365-files": ("FILES.json", "VERSION"),
}

# Upstream sources: repo, commit, stage dir, mirror dir.
_SOURCES = {
    "robocasa-src": (
        "robocasa/robocasa",
        "756598a5be52e052339bb2d957426e39015c2afb",
        "robocasa-v02-git",
    ),
    "robocasa365-src": (
        "robocasa/robocasa",
        "4f8a2980def75a55dff96b990745b83540425f09",
        "robocasa-101-git",
    ),
    "robocasa-gr1-src": (
        "robocasa/robocasa-gr1-tabletop-tasks",
        "4840e671596f93ca03651524b9f72ffb1aadfeff",
        "robocasa-gr1-git",
    ),
    # robosuite @ 51cc017 is the v1.5.1 release point (changelog merge); the
    # v0.2 and GR1 images both run robosuite 1.5.1, so one mirror serves both.
    "robosuite-151-src": (
        "ARISE-Initiative/robosuite",
        "51cc01785bab80ffeed20da15e67d7dd4140e76a",
        "robosuite-151-git",
    ),
    "robosuite-152-src": (
        "ARISE-Initiative/robosuite",
        "5ce6643f3092639d08f7b0f90ed1c6a84f50552c",
        "robosuite-152-git",
    ),
}

_HF = "https://hf-mirror.com"
ROBOCASA_ASSETS = f"{_HF}/datasets/robocasa/robocasa-assets/resolve/1b92c3d02ca4354984fec961357db0bff7b32166"
NV_ASSETS = f"{_HF}/datasets/nvidia/PhysicalAI-DigitalCousin-Assets/resolve/1b018839a6da865dffecd3185fe054211bc71270"
V02_ASSETS = f"{_HF}/datasets/jianzhang96/robocasa-assets/resolve/866be1d2158a6486af2211156b83eeaeb031f54f"


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-robocasa {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run git with the local proxy explicitly disabled."""
    env = os.environ.copy()
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(var, None)
    return subprocess.run(
        ["git", "-c", "http.proxy=", "-c", "https.proxy=", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def git_show(repo: Path, ref: str, rel: str) -> bytes | None:
    """Return a file's bytes from a git checkout, or None."""
    proc = git("show", f"{ref}:{rel}", cwd=repo)
    if proc.returncode != 0:
        return None
    return proc.stdout.encode()


def stage_dir() -> Path:
    """Locate the staged EFE checkout holding the robocasa task dirs."""
    for candidate in EFE_STAGE_CANDIDATES:
        if (candidate / "tasks" / "robocasa").is_dir():
            return candidate
    raise SystemExit(
        "staged EFE checkout with tasks/robocasa not found; run the sparse "
        "checkout first (git -C .efe_stage/<efe-repo> sparse-checkout add "
        "tasks/robocasa tasks/robocasa365 tasks/robocasa-gr1)"
    )


def mirror_files(
    src_root: Path, dst_root: Path, rel_paths: list[str]
) -> tuple[int, int]:
    """Copy rel_paths from src to dst; skip identical existing files."""
    copied = skipped = 0
    for rel in rel_paths:
        src, dst = src_root / rel, dst_root / rel
        if not src.is_file():
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def mirror_efe_tasks() -> None:
    """Mirror the three suites' task dirs and the image build recipes."""
    stage = stage_dir()
    total_c = total_s = 0
    for suite in SUITES:
        tasks_src = stage / "tasks" / suite
        if not tasks_src.is_dir():
            log(f"WARNING: {tasks_src} missing; skipped")
            continue
        rels = [f"{suite}/README.md"]
        for task_dir in sorted(p for p in tasks_src.iterdir() if p.is_dir()):
            for name in _TASK_FILES:
                if (task_dir / name).is_file():
                    rels.append(f"{suite}/{task_dir.name}/{name}")
        copied, skipped = mirror_files(stage / "tasks", UPSTREAM_OUT / "tasks", rels)
        total_c += copied
        total_s += skipped
        log(f"EFE tasks/{suite}: {copied} copied, {skipped} present")
    for image, names in _IMAGE_FILES.items():
        src = stage / "images" / image
        if not src.is_dir():
            log(f"WARNING: images/{image} missing; skipped")
            continue
        rels = [n for n in names if (src / n).is_file()]
        copied, skipped = mirror_files(src, IMAGES_OUT / image, rels)
        total_c += copied
        total_s += skipped
    log(f"EFE mirror done: {total_c} copied, {total_s} present")


def ensure_source(key: str) -> Path | None:
    """Clone/update one upstream source mirror; return its dir or None.

    Reuses a ready stage clone (``.efe_stage/<stage>-git``) produced by an
    earlier manual step; otherwise clones from GitHub (ghfast.top relay as
    fallback). The mirror drops ``.git`` and records the pinned commit in
    ``REF`` — provenance lives in this tool's constants.
    """
    repo, ref, stage_name = _SOURCES[key]
    out = UPSTREAM_OUT / key
    ref_file = out / "REF"
    if ref_file.is_file() and ref_file.read_text(encoding="utf-8").strip() == ref:
        return out
    stage = STAGE / stage_name
    if not (stage / ".git").is_dir():
        urls = [f"https://github.com/{repo}.git"]
        if not repo.startswith("https://ghfast.top/"):
            urls.append(f"https://ghfast.top/https://github.com/{repo}.git")
        for url in urls:
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                proc = git("clone", "--quiet", "--filter=blob:none", url, str(out))
                if proc.returncode != 0:
                    log(f"clone {url} failed: {(proc.stderr or '').strip()[-200:]}")
                    continue
                git("fetch", "--quiet", "--depth", "1", "origin", ref, cwd=out)
                git("checkout", "--quiet", ref, cwd=out)
                break
            except subprocess.TimeoutExpired:
                log(f"timeout cloning {url}")
        else:
            log(f"WARNING: could not clone {repo} @ {ref[:8]}")
            return None
    else:
        proc = git("checkout", "--quiet", ref, cwd=stage)
        if proc.returncode != 0:
            log(f"WARNING: stage {stage} cannot checkout {ref[:8]}")
            return None
        shutil.rmtree(out, ignore_errors=True)
        ignore = shutil.ignore_patterns(".git")
        shutil.copytree(stage, out, ignore=ignore)
    shutil.rmtree(out / ".git", ignore_errors=True)
    ref_file.write_text(ref + "\n", encoding="utf-8")
    log(f"source {key} @ {ref[:8]} ready")
    return out


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_length(url: str) -> int:
    """Total size of a (redirected) HTTP resource."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_parallel(
    url: str,
    dest: Path,
    sha256: str,
    *,
    chunk_size: int = 8 << 20,
    workers: int = 16,
) -> str:
    """Range-parallel downloader with per-chunk resume (throttled mirrors).

    Single connections to the mirror throttle to ~50-230 KB/s; N parallel
    range requests scale the aggregate roughly linearly. Chunks land in
    ``<dest>.parts/`` and are assembled + SHA-256 verified only when every
    chunk is present, so re-runs continue where a killed run stopped (the
    finished ``dest`` with a matching hash is never re-downloaded).
    """
    if dest.is_file() and sha256_of(dest) == sha256:
        return "present"
    total = _content_length(url)
    parts = dest.with_suffix(dest.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    n_chunks = (total + chunk_size - 1) // chunk_size

    def fetch(index: int) -> str:
        chunk = parts / f"{index:06d}"
        expect = min(chunk_size, total - index * chunk_size)
        if chunk.is_file() and chunk.stat().st_size == expect:
            return "skip"
        tmp = chunk.with_suffix(".tmp")
        attempt = 0
        while time.time() < DEADLINE:
            attempt += 1
            try:
                request = urllib.request.Request(url)
                start = index * chunk_size
                request.add_header("Range", f"bytes={start}-{start + expect - 1}")
                with urllib.request.urlopen(request, timeout=300) as response:
                    if response.status != 206:
                        raise OSError(f"no partial content: {response.status}")
                    with tmp.open("wb") as handle:
                        shutil.copyfileobj(response, handle, 1 << 20)
                if tmp.stat().st_size == expect:
                    tmp.replace(chunk)
                    return "ok"
                tmp.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - network is flaky
                wait = min(30, 2**attempt)
                log(f"chunk {index} attempt {attempt}: {exc}; wait {wait}s")
                time.sleep(wait)
        raise TimeoutError(f"chunk {index} failed after {attempt} attempts")

    from concurrent.futures import ThreadPoolExecutor

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(fetch, range(n_chunks)):
            done += 1
            if status == "ok" or done % 25 == 0:
                log(f"{dest.name}: chunks {done}/{n_chunks}")
    assembled = dest.with_suffix(dest.suffix + ".assembled")
    with assembled.open("wb") as out:
        for index in range(n_chunks):
            chunk = parts / f"{index:06d}"
            expect = min(chunk_size, total - index * chunk_size)
            if not chunk.is_file() or chunk.stat().st_size != expect:
                raise OSError(f"chunk {index} missing after assembly scan")
            with chunk.open("rb") as handle:
                shutil.copyfileobj(handle, out, 1 << 20)
    if assembled.stat().st_size != total:
        raise OSError(f"assembled size mismatch: {assembled.stat().st_size} != {total}")
    if sha256_of(assembled) != sha256:
        assembled.unlink()
        raise OSError("assembled sha256 mismatch")
    assembled.replace(dest)
    shutil.rmtree(parts, ignore_errors=True)
    return "downloaded"


def download(url: str, dest: Path, sha256: str) -> str:
    """Single-connection resumable download; see download_parallel for bulk."""
    if dest.is_file() and sha256_of(dest) == sha256:
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while time.time() < DEADLINE:
        attempt += 1
        try:
            request = urllib.request.Request(url)
            if part.is_file():
                request.add_header("Range", f"bytes={part.stat().st_size}-")
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if response.status == 206 else "wb"
                if response.status == 200 and part.is_file():
                    part.unlink()  # server ignored the Range header
                    mode = "wb"
                with part.open(mode) as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            if sha256_of(part) == sha256:
                part.replace(dest)
                return "downloaded"
            log(f"sha256 mismatch for {dest.name}; retrying ({attempt})")
            part.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - network is flaky by design
            wait = min(60, 2**attempt)
            log(f"download {dest.name} attempt {attempt} failed: {exc}; wait {wait}s")
            time.sleep(wait)
    return "failed"


# Dead UT Box archives with no confirmed mirror; recorded for the probe step.
MISSING = {
    "robocasa-src": (
        "objects UT Box 两档（ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip / "
        "os3hrui06lasnuvwqpmwn0wcrduh6jg3.zip）→ 由 HF objaverse/aigen 替代"
    ),
    "robocasa365-src": (
        "idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip（$assets 根）与 "
        "vckqvvkh1z8t69k8qcpcmee6k66stii4.zip（objects）无镜像"
    ),
    "robocasa-gr1-src": (
        "otdsyfjontk17jdp24bkhy2hgalofbh4.zip 与 pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip"
        "（GR1 tabletop 场景固件）无镜像"
    ),
}


def suite_references(suite: str) -> set[str]:
    """XML-referenced asset paths (relative to models/assets) for a suite."""
    import numpy as np  # local: this tool runs under uv, tests never call it

    refs: set[str] = set()
    pattern = f"data/efe/upstream/tasks/{suite}/*-privileged/environment/instance.npz"
    for npz in sorted(ROOT.glob(pattern)):
        with zipfile.ZipFile(npz) as archive:
            xml = np.load(archive.open("xml.npy"), allow_pickle=True).item()
        for match in re.finditer(r'file="(/[^"]+)"', xml):
            path = match.group(1)
            marker = "models/assets/"
            if marker in path:
                refs.add(path.split(marker, 1)[1])
    return refs


#: suite -> (robosuite mirror key, archives for the root, archives for objects/)
# (url, sha256) for every pinned asset archive.
_ASSET_ARCHIVES = {
    "textures": (
        f"{ROBOCASA_ASSETS}/textures.zip",
        "ba3a4dcef96e199cbd601c7cdb7eb46077fb6672611b7ac74c25a191bf0be392",
    ),
    "fixtures": (
        f"{ROBOCASA_ASSETS}/fixtures.zip",
        "e7bb86e9e0ef130de78200a0a7465dfb0161ed3956c65d8b3f36bdff3746c7a8",
    ),
    "fixtures_v02": (
        f"{V02_ASSETS}/fixtures.zip",
        "bd993f47d0ae5f5f3d89cf5f9fc28d55ceb878359c49136d30052de0bb1ef231",
    ),
    "generative_textures": (
        f"{ROBOCASA_ASSETS}/generative_textures.zip",
        "bc05095af306222d9a799a9b0b5b16662aa546998622ee8fd3072566ea4c31ec",
    ),
    "objaverse": (
        f"{ROBOCASA_ASSETS}/objaverse.zip",
        "66a7eebef3bac855301964d35d905fc2cb44d691f007f97af08d96e83db5cc08",
    ),
    "aigen_objs": (
        f"{ROBOCASA_ASSETS}/aigen_objs.zip",
        "1f88792322ccce79f775e88482626c324c6e07d002adf23320a124b72764dd19",
    ),
    "lightwheel": (
        f"{NV_ASSETS}/lightwheel.zip",
        "8e42a0c835a205d7a0e7c33b5c33ab160b220fa9343b50c16526b7c712a2b6e2",
    ),
    "sketchfab": (
        f"{NV_ASSETS}/sketchfab.zip",
        "12af71e68a5c748f8531ce20c9db0ae9ed71ab1d46265388e05155a5b5352b5d",
    ),
}

_SUITE_POPULATE = {
    "robocasa": {
        "robosuite": "robosuite-151-src",
        "root": ("textures", "fixtures", "fixtures_v02"),
        "objects": ("objaverse",),
    },
    "robocasa365": {
        "robosuite": "robosuite-152-src",
        "root": ("textures", "generative_textures"),
        "objects": ("lightwheel", "objaverse", "aigen_objs"),
    },
    "robocasa-gr1": {
        "robosuite": "robosuite-151-src",
        "root": ("textures", "fixtures_v02"),
        "objects": ("lightwheel", "sketchfab", "objaverse"),
    },
}

_RS_TREE = {
    "robosuite-151-src": UPSTREAM_OUT
    / "robosuite-151-src"
    / "robosuite"
    / "models"
    / "assets",
    "robosuite-152-src": UPSTREAM_OUT
    / "robosuite-152-src"
    / "robosuite"
    / "models"
    / "assets",
}


def mirror_assets() -> None:
    """Download the pinned archives and populate each suite's unified root.

    The unified root (``assets/robocasa/<suite>/models/assets``) receives
    only the members the suite's world XMLs reference — from the HF/NVIDIA
    zips and from the suite's robosuite source mirror — so the runtime
    spawn touches nothing else.
    """
    zips = UPSTREAM_OUT / "robocasa-asset-zips"
    wanted: dict[str, tuple[str, str]] = {}
    for groups in _SUITE_POPULATE.values():
        for names in (groups["root"], groups["objects"]):
            for name in names:
                wanted[name] = _ASSET_ARCHIVES[name]
    for name, (url, sha256) in wanted.items():
        archive = zips / f"{name}.zip"
        status = download(url, archive, sha256)
        if status == "failed":
            log(f"FAILED archive {name}")
        else:
            log(f"archive {name}: {status} ({archive.stat().st_size >> 20} MB)")
    for suite, groups in _SUITE_POPULATE.items():
        root = ROOT / "assets" / "robocasa" / suite / "models" / "assets"
        refs = suite_references(suite)
        for sub, names in (("root", groups["root"]), ("objects", groups["objects"])):
            for name in names:
                archive = zips / f"{name}.zip"
                if not archive.is_file():
                    continue
                dest_root = root if sub == "root" else root / "objects"
                # Archives carry their own top level (objaverse.zip holds
                # ``objaverse/<cat>/...`` and is unpacked into ``objects/``),
                # so members match reference paths with the destination
                # prefix stripped.
                prefix = "" if sub == "root" else "objects/"
                wanted = {r[len(prefix) :] for r in refs if r.startswith(prefix)}
                # The object-library MJCF (<obj>/model.xml) is loaded by
                # robocasa code, never referenced by the world XML's file=
                # attributes — extract it alongside the referenced members
                # so radii/bbox baking finds it.
                dirs = {"/".join(w.split("/")[:3]) + "/model.xml" for w in wanted}
                members = {
                    n for n in zipfile.ZipFile(archive).namelist() if n in wanted
                } | {n for n in dirs if n in zipfile.ZipFile(archive).namelist()}
                if not members:
                    log(f"extract {name} -> {suite}/{sub}: 0 members")
                    continue
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(dest_root, members=members)
                log(f"extract {name} -> {suite}/{sub}: {len(members)} members")
        rs_root = _RS_TREE[groups["robosuite"]]
        if rs_root.is_dir():
            copied = 0
            for rel in sorted(refs):
                src = rs_root / rel
                dst = root / rel
                if src.is_file() and not dst.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1
            log(f"robosuite {groups['robosuite']} -> {suite}: {copied} files")
    for key, note in MISSING.items():
        log(f"MISSING {key}: {note}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tasks-only", action="store_true")
    group.add_argument("--assets-only", action="store_true")
    group.add_argument("--sources-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.assets_only:
        mirror_efe_tasks()
    if not args.tasks_only:
        for key in _SOURCES:
            ensure_source(key)
        if not args.sources_only:
            mirror_assets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
