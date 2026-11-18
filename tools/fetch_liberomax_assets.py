#!/usr/bin/env python3
"""Repo-local, idempotent mirror of LIBERO-MAX task assets for the EFE port.

LIBERO-MAX (EFE suite ``libero-max``) ships only per-case ``instance.json``
files in the EFE repo: each frozen instance references LIBERO-PRO task data
under Hugging Face ``zhouxueyang/LIBERO-Pro`` (BDDL + ``.pruned_init``
initial-state arrays), the LIBERO package and object/scene assets from
``Zxy-MLlab/LIBERO-PRO`` (MIT), and LIBERO-MAX's event runtime from
``liberomax/LIBERO-MAX`` @ ``a1e3cef`` (redistributed with its authors'
permission, see EFE ``tasks/libero-max/README.md``). This tool mirrors
exactly what the 64 registered cases need:

* EFE task files (``instance.json``, ``instruction.md``, ``task.toml``,
  ``solution/oracle.npy``) -> ``data/efe/upstream/tasks/libero-max/``
  (mll-lab-nu/embodied-first-exam @ main); the image build recipes go to
  ``data/efe/upstream/images-libero-max/``;
* LIBERO-MAX event runtime (``src/libero_max`` + ``scripts/libero_source_overlay``)
  -> ``data/efe/upstream/libero-max-src/`` (liberomax/LIBERO-MAX @ a1e3cef);
* LIBERO-PRO source without its assets subtree
  -> ``data/efe/upstream/libero-pro-src/``; the assets subtree
  (scenes / objects / textures referenced by MJCF relative paths)
  -> ``assets/liberomax/`` (Zxy-MLlab/LIBERO-PRO @ 2b910b5, MIT);
* BDDL + ``.pruned_init`` files for the registered cases plus the dataset
  manifest and SHA256SUMS -> ``data/efe/upstream/libero-pro-data/``
  (zhouxueyang/LIBERO-Pro @ c86fc3b, CC-BY-4.0). Downloaded files are
  verified against the mirrored SHA256SUMS.txt.

Like ``tools/fetch_vlabench_assets.py`` the mirrors stay local and
uncommitted; only this tool is. CI and tests never call it. Network is
flaky by design: per-request retries with backoff, skip files already
present and non-empty, re-runs only fill gaps. Git access tries GitHub
directly first and the ghfast.top relay as fallback, both with the local
proxy explicitly disabled.

Usage::

    uv run python tools/fetch_liberomax_assets.py
    HF_ENDPOINT=https://hf-mirror.com uv run python tools/fetch_liberomax_assets.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = time.time() + 90 * 60

EFE_REPO = "mll-lab-nu/embodied-first-exam"
LMX_REPO = "liberomax/LIBERO-MAX"
LMX_REF = "a1e3cef258b3e00487db5282aef4e36010aaf401"
PRO_REPO = "Zxy-MLlab/LIBERO-PRO"
PRO_REF = "2b910b5b5f53016bef9907632f6f840f1ce2229c"
HF_REPO = "zhouxueyang/LIBERO-Pro"
HF_REV = "c86fc3b8293185a6f373677018ff3e37f8391602"

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream"
TASKS_OUT = UPSTREAM_OUT / "tasks" / "libero-max"
IMAGES_OUT = UPSTREAM_OUT / "images-libero-max"
LMX_SRC_OUT = UPSTREAM_OUT / "libero-max-src"
PRO_SRC_OUT = UPSTREAM_OUT / "libero-pro-src"
DATA_OUT = UPSTREAM_OUT / "libero-pro-data"
ASSETS_OUT = ROOT / "assets" / "liberomax"

# huggingface.co first, the mirror as fallback (both may be unreachable;
# whichever answers wins). HF_ENDPOINT overrides both.
_HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

# EFE task files worth mirroring (relative to each task dir).
_TASK_FILES = (
    "environment/sim/instance.json",
    "environment/instance.json",
    "instruction.md",
    "task.toml",
    "README.md",
    "solution/oracle.npy",
    "solution/solve.sh",
    "solution/oracle_replay.py",
)

_METADATA_FILES = (
    "metadata/bddl_40task_7perturbation_manifest.json",
    "metadata/dataset_index.json",
    "metadata/static_validation_40task_7perturbation_summary.json",
    "SHA256SUMS.txt",
)


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[fetch-liberomax {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run git with the local proxy explicitly disabled."""
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    return subprocess.run(
        ["git", "-c", "http.proxy=", "-c", "https.proxy=", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def clone_sparse(repo: str, ref: str, patterns: list[str], out: Path) -> Path:
    """Sparse-clone ``repo`` at ``ref`` into ``out``; return the checkout."""
    """Sparse-clone ``repo`` at ``ref`` into ``out``; return the git dir.

    Tries GitHub directly, then the ghfast.top relay. Existing checkouts are
    reused when they already contain ``ref``.
    """
    if (out / ".git").is_dir():
        proc = git("cat-file", "-e", ref, cwd=out)
        if proc.returncode == 0:
            log(f"reuse {out} (has {ref[:8]})")
            return out
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    urls = [f"https://github.com/{repo}.git"]
    if not repo.startswith("https://ghfast.top/"):
        urls.append(f"https://ghfast.top/https://github.com/{repo}.git")
    last_err = ""
    for url in urls:
        try:
            if (out / ".git").is_dir():
                git("remote", "set-url", "origin", url, cwd=out)
                proc = git("fetch", "--depth", "1", "origin", ref, cwd=out)
            else:
                shutil.rmtree(out, ignore_errors=True)
                proc = git(
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    "--filter=blob:none",
                    "--depth",
                    "1",
                    url,
                    str(out),
                    cwd=out.parent,
                )
                if proc.returncode == 0:
                    proc = git("fetch", "--depth", "1", "origin", ref, cwd=out)
            if proc.returncode != 0:
                last_err = (proc.stderr or proc.stdout).strip()[-300:]
                log(f"fetch {url} failed: {last_err}")
                continue
            git("sparse-checkout", "init", "--no-cone", cwd=out)
            git("sparse-checkout", "set", *patterns, cwd=out)
            proc = git("checkout", "-q", "FETCH_HEAD", cwd=out)
            if proc.returncode != 0:
                last_err = (proc.stderr or proc.stdout).strip()[-300:]
                log(f"checkout {url} failed: {last_err}")
                continue
            log(f"cloned {repo} @ {ref[:8]} -> {out}")
            return out
        except subprocess.TimeoutExpired:
            last_err = f"timeout cloning {url}"
            log(last_err)
    raise RuntimeError(f"cannot clone {repo} @ {ref}: {last_err}")


def mirror_files(
    src_root: Path, dst_root: Path, rel_paths: list[str]
) -> tuple[int, int]:
    """Copy ``rel_paths`` from src to dst; skip identical existing files."""
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
    """Mirror EFE libero-max task dirs and image recipes."""
    stage = ROOT / ".efe_stage" / "efe-git"
    clone_sparse(
        EFE_REPO,
        "main",
        [
            "/tasks/libero-max/",
            "/images/libero-max/",
            "/tasks/README.md",
        ],
        stage,
    )
    tasks_src = stage / "tasks" / "libero-max"
    task_dirs = sorted(p for p in tasks_src.iterdir() if p.is_dir())
    rels = []
    for task_dir in task_dirs:
        for name in _TASK_FILES:
            if (task_dir / name).is_file():
                rels.append(f"{task_dir.name}/{name}")
    copied, skipped = mirror_files(tasks_src, TASKS_OUT, rels)
    img_rels = [
        p.name for p in (stage / "images" / "libero-max").iterdir() if p.is_file()
    ]
    copied2, skipped2 = mirror_files(
        stage / "images" / "libero-max", IMAGES_OUT, img_rels
    )
    log(
        f"EFE tasks: {copied + copied2} copied, {skipped + skipped2} present "
        f"({len(task_dirs)} task dirs)"
    )


def mirror_libero_max_src() -> None:
    """Mirror LIBERO-MAX's event runtime source."""
    stage = clone_sparse(
        LMX_REPO,
        LMX_REF,
        ["/src/", "/scripts/libero_source_overlay/", "/LICENSE", "/README.md"],
        ROOT / ".efe_stage" / "libero-max-git",
    )
    copied, skipped = mirror_files(
        stage,
        LMX_SRC_OUT,
        [
            str(p.relative_to(stage)).replace(os.sep, "/")
            for p in stage.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ],
    )
    log(f"LIBERO-MAX source: {copied} copied, {skipped} present")


def mirror_libero_pro() -> None:
    """Mirror LIBERO-PRO source and its assets subtree."""
    stage = clone_sparse(
        PRO_REPO,
        PRO_REF,
        ["/libero/", "/LICENSE", "/README.md", "/setup.py", "/requirements.txt"],
        ROOT / ".efe_stage" / "libero-pro-git",
    )
    # The source mirror intentionally keeps the assets subtree: LIBERO object
    # classes resolve their MJCF relative to the *source tree*
    # (libero/libero/assets/...), independent of the configured assets path.
    # assets/liberomax carries the identical bytes for compiled configs.
    src_files = [
        str(p.relative_to(stage)).replace(os.sep, "/")
        for p in stage.rglob("*")
        if p.is_file() and ".git" not in p.parts
    ]
    copied, skipped = mirror_files(stage, PRO_SRC_OUT, src_files)
    log(f"LIBERO-PRO source: {copied} copied, {skipped} present")

    assets_stage = stage / "libero" / "libero" / "assets"
    marker = ASSETS_OUT / ".mirror-complete"
    if marker.is_file():
        log(f"reuse assets mirror at {ASSETS_OUT}")
    else:
        ASSETS_OUT.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in assets_stage.rglob("*"):
            if p.is_file():
                rel = p.relative_to(assets_stage)
                dst = ASSETS_OUT / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                n += 1
        marker.write_text(f"{n} files from {PRO_REPO} @ {PRO_REF}\n", encoding="utf-8")
        log(f"LIBERO-PRO assets: {n} files -> {ASSETS_OUT}")


def case_file_lists() -> tuple[set[str], set[str]]:
    """Collect the BDDL/init files every registered instance references."""
    bddl, init = set(), set()
    for inst in sorted(TASKS_OUT.glob("*-standard/environment/sim/instance.json")):
        case = json.loads(inst.read_text(encoding="utf-8"))
        variant = case["case"]["substrate_variant"]
        bddl.add(variant["bddl_file"])
        init.add(variant["init_states_file"])
        ref = variant.get("init_reference_bddl_file")
        if ref:
            bddl.add(ref)
    return bddl, init


def hf_endpoints() -> list[str]:
    """Resolve-order Hugging Face endpoints (HF_ENDPOINT overrides)."""
    override = os.environ.get("HF_ENDPOINT")
    if override:
        return [override.rstrip("/")]
    return list(_HF_ENDPOINTS)


def hf_fetch(rel_path: str, dst: Path, endpoints: list[str]) -> None:
    """Download one dataset file with retries; skip when present."""
    if dst.is_file() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(6):
        if time.time() > DEADLINE:
            raise TimeoutError(f"deadline exceeded while fetching {rel_path}")
        base = endpoints[attempt % len(endpoints)]
        url = (
            f"{base}/datasets/{HF_REPO}/resolve/{HF_REV}/{urllib.parse.quote(rel_path)}"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "genesis-cloud-sim-fetch"}
            )
            with (
                urllib.request.urlopen(req, timeout=120) as resp,
                open(dst, "wb") as fh,
            ):
                shutil.copyfileobj(resp, fh)
            return
        except Exception as exc:  # noqa: BLE001 - network flakiness by design
            last_err = str(exc)
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"cannot fetch {rel_path}: {last_err}")


def verify_sha256sums() -> None:
    """Verify mirrored files against the dataset's SHA256SUMS.txt."""
    sums_file = DATA_OUT / "SHA256SUMS.txt"
    if not sums_file.is_file():
        return
    wanted = set()
    bddl, init = case_file_lists()
    wanted |= {f"bddl_files/{p}" for p in bddl}
    wanted |= {f"init_files/{p}" for p in init}
    bad = []
    checked = 0
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        digest, rel = parts
        rel = rel.lstrip("*")
        if rel not in wanted:
            continue
        dst = DATA_OUT / rel
        if not dst.is_file():
            continue
        checked += 1
        if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            bad.append(rel)
            dst.unlink()
    if bad:
        raise RuntimeError(f"sha256 mismatch, removed for re-fetch: {bad}")
    log(f"sha256 verified: {checked} files OK")


def mirror_hf_data() -> None:
    """Mirror the BDDL/init/manifest files the registered cases need."""
    endpoints = hf_endpoints()
    bddl, init = case_file_lists()
    rels = (
        [f"bddl_files/{p}" for p in sorted(bddl)]
        + [f"init_files/{p}" for p in sorted(init)]
        + list(_METADATA_FILES)
    )
    todo = [r for r in rels if not (DATA_OUT / r).is_file()]
    log(f"HF data: {len(todo)}/{len(rels)} files to fetch")
    if todo:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda r: hf_fetch(r, DATA_OUT / r, endpoints), todo))
    verify_sha256sums()
    log(f"HF data mirrored -> {DATA_OUT}")


def main() -> int:
    """Mirror every LIBERO-MAX source; entry point."""
    mirror_efe_tasks()
    mirror_libero_max_src()
    mirror_libero_pro()
    mirror_hf_data()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
