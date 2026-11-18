#!/usr/bin/env python3
"""Repo-local, idempotent mirror of VLABench task assets for the EFE P1 port.

VLABench (EFE suite ``vlabench``) ships no task XML in the EFE repo: each
frozen instance references MJCF components under Hugging Face
``lerobot/vlabench-assets`` and the Panda robot under the upstream GitHub
repo. This tool mirrors exactly the files the 36 registered tasks need:

* EFE task files (``environment/instance.json``, ``instruction.md``,
  ``tests/test.sh``/``instance.json``, ``solution/oracle.npy``) -> ``data/efe/upstream/tasks/vlabench/``
  (mll-lab-nu/embodied-first-exam @ main); the verifier source (``grade.py``)
  lives under ``images/vlabench/`` and is mirrored to
  ``data/efe/upstream/images-vlabench/``;
* component MJCF + referenced meshes/textures -> ``assets/vlabench/obj/``
  (HF dataset ``lerobot/vlabench-assets`` @ 08d7a44);
* scene MJCF + referenced assets -> ``assets/vlabench/scenes/``;
* Panda robot MJCF + meshes -> ``assets/vlabench/robots/franka_emika_panda/``
  (OpenMOSS/VLABench @ cf588fe60c0c7282174fe979f5913170cfe69017).

VLABench's asset set has no licence that allows redistribution, so the
mirror stays local and gitignored; only this tool is committed. Manual
step, like ``tools/fetch_upstream_assets.py`` — CI and tests never call it.
Network is flaky by design: per-request retries with backoff, skip files
already present and non-empty, re-runs only fill gaps.

Usage::

    uv run python tools/fetch_vlabench_assets.py
    HF_ENDPOINT=https://hf-mirror.com uv run python tools/fetch_vlabench_assets.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = time.time() + 40 * 60

EFE_REPO = "mll-lab-nu/embodied-first-exam"
VLB_REPO = "OpenMOSS/VLABench"
VLB_REF = "cf588fe60c0c7282174fe979f5913170cfe69017"
HF_REPO = "lerobot/vlabench-assets"
HF_REV = "08d7a44"
PANDA_DIR = "VLABench/assets/robots/franka_emika_panda"

UPSTREAM_OUT = ROOT / "data" / "efe" / "upstream" / "tasks" / "vlabench"
ASSETS_OUT = ROOT / "assets" / "vlabench"

# huggingface.co first, the mirror as fallback (both may be unreachable;
# whichever answers wins). HF_ENDPOINT overrides both.
_HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]


def hf_endpoints() -> list[str]:
    """HF dataset endpoints, ``HF_ENDPOINT`` override first."""
    override = os.environ.get("HF_ENDPOINT")
    if override:
        return [override.rstrip("/")]
    return _HF_ENDPOINTS


def get(
    url: str, dest: Path | None = None, tries: int = 8, max_time: int = 120
) -> bool:
    """Fetch a URL to a path (or discard the body) with retry/backoff."""
    for attempt in range(tries):
        try:
            if dest is None:
                subprocess.run(
                    [
                        "curl",
                        "-sfL",
                        "--max-time",
                        str(max_time),
                        "-o",
                        os.devnull,
                        url,
                    ],
                    check=True,
                    capture_output=True,
                )
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                subprocess.run(
                    ["curl", "-sfL", "--max-time", str(max_time), "-o", str(tmp), url],
                    check=True,
                    capture_output=True,
                )
                tmp.replace(dest)
            return True
        except subprocess.CalledProcessError:
            time.sleep(min(1.5**attempt, 15))
    return False


def gh_raw(repo: str, ref: str, path: str, dest: Path) -> str:
    """Fetch one GitHub file into the mirror when missing.

    raw.githubusercontent may be unreachable while api.github.com still
    answers; fall back to the contents API (raw media type, small files).
    """
    if dest.is_file() and dest.stat().st_size > 0:
        return "have"
    if get(f"https://raw.githubusercontent.com/{repo}/{ref}/{path}", dest):
        return "ok"
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    for _ in range(3):
        try:
            out = subprocess.run(
                [
                    "curl",
                    "-sfL",
                    "--max-time",
                    "60",
                    "-H",
                    "Accept: application/vnd.github.raw",
                    url,
                ],
                check=True,
                capture_output=True,
            )
            if out.stdout:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(out.stdout)
                return "ok"
        except subprocess.CalledProcessError:
            time.sleep(2)
    return f"MISS {path}"


def gh_mirror_prefix(repo: str, ref: str, prefix: str, out_dir: Path) -> str:
    """Sparse-clone ``prefix`` of ``repo`` at ``ref`` into ``out_dir``.

    git's smart-HTTP transfer survives flaky links far better than codeload
    tarballs (which reject HTTP ranges and cannot resume). Blobless partial
    clone fetches only the blobs under ``prefix``. Idempotent: files already
    present and non-empty are never overwritten.
    """
    import shutil

    stage = ROOT / ".efe_stage" / f"{repo.replace('/', '_')}-git"
    url = f"https://github.com/{repo}.git"

    def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[bytes]:
        # git may carry a stale local-proxy config (e.g. 127.0.0.1:7897) while
        # the direct route works; retry with the proxy disabled on failure.
        proc = subprocess.run(["git", *args], capture_output=True)
        if check and proc.returncode != 0:
            proc = subprocess.run(
                ["git", "-c", "http.proxy=", "-c", "https.proxy=", *args],
                capture_output=True,
            )
            proc.check_returncode()
        return proc

    try:
        if not (stage / ".git").exists():
            _git(
                [
                    "clone",
                    "--no-checkout",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    url,
                    str(stage),
                ]
            )
        _git(["-C", str(stage), "sparse-checkout", "set", prefix])
        pinned = "HEAD"  # clone's default tip; overridden when the ref fetch works
        fetch = subprocess.run(
            ["git", "-C", str(stage), "fetch", "--depth", "1", "origin", ref],
            capture_output=True,
        )
        if fetch.returncode == 0:
            pinned = "FETCH_HEAD"
        subprocess.run(
            ["git", "-C", str(stage), "checkout", pinned],
            check=True,
            capture_output=True,
        )
        src = stage / prefix
        n = 0
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            dest = out_dir / f.relative_to(src)
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            n += 1
        return f"clone ok (+{n} new, ref={pinned})"
    except subprocess.CalledProcessError as e:
        return f"clone FAIL rc={e.returncode} {e.stderr.decode(errors='replace')[:120]}"


_HF_ALIVE: list[str] | None = None


def _hf_alive() -> list[str]:
    """Endpoints that answer, probed once (a dead host costs minutes per file)."""
    global _HF_ALIVE
    if _HF_ALIVE is None:
        alive = []
        for ep in hf_endpoints():
            rc = subprocess.run(
                [
                    "curl",
                    "-sfL",
                    "--max-time",
                    "15",
                    "-o",
                    os.devnull,
                    f"{ep}/api/datasets/{HF_REPO}",
                ],
                capture_output=True,
            ).returncode
            if rc == 0:
                alive.append(ep)
        _HF_ALIVE = alive or hf_endpoints()[-1:]
        print(f"[hf] endpoints alive: {_HF_ALIVE}", flush=True)
    return _HF_ALIVE


def hf(path: str, dest: Path) -> str:
    """Fetch one HF dataset file, trying each live endpoint until one answers."""
    if dest.is_file() and dest.stat().st_size > 0:
        return "have"
    for ep in _hf_alive():
        if get(
            f"{ep}/datasets/{HF_REPO}/resolve/{HF_REV}/{path}",
            dest,
            tries=3,
            max_time=90,
        ):
            return "ok"
    return f"MISS {path}"


def hf_list(path: str) -> list[dict] | None:
    """List one tree level of the HF dataset via the API (any live endpoint)."""
    for ep in _hf_alive():
        for _ in range(4):
            try:
                out = subprocess.run(
                    [
                        "curl",
                        "-sfL",
                        "--max-time",
                        "60",
                        f"{ep}/api/datasets/{HF_REPO}/tree/{HF_REV}/{path}",
                    ],
                    check=True,
                    capture_output=True,
                )
                return json.loads(out.stdout)
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                time.sleep(3)
    return None


def gh_tree(repo: str, ref: str, path: str) -> list[dict] | None:
    """List one git tree level via the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}"
    if path:
        url += f":{path}"
    for _ in range(6):
        try:
            out = subprocess.run(
                ["curl", "-sfL", "--max-time", "60", url],
                check=True,
                capture_output=True,
            )
            return json.loads(out.stdout)["tree"]
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            time.sleep(4)
    return None


def xml_file_refs(xml_path: Path) -> list[str]:
    """Repo-relative paths of the mesh/texture files an MJCF references.

    MuJoCo resolves ``<mesh>`` against ``compiler meshdir`` and ``<texture>``
    against ``texturedir`` (falling back to ``meshdir``); both default to the
    model's own directory. Returns paths relative to the mirror root.
    """
    import posixpath

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else None) or ""
    # MuJoCo resolves textures against ``texturedir`` which defaults to the
    # model's own directory (NOT meshdir) — the fridge assets rely on this.
    texturedir = (compiler.get("texturedir") if compiler is not None else None) or ""
    refs = []
    for elem in root.iter():
        f = elem.attrib.get("file")
        if not f or elem.tag not in ("mesh", "texture", "hfield", "include"):
            continue
        sub = texturedir if elem.tag == "texture" else meshdir
        refs.append(posixpath.normpath(posixpath.join(sub, f)))
    return refs


def fetch_efe_tasks() -> None:
    """Mirror instruction/instance files for the 36 registered tasks.

    ``tests/grade.py`` is NOT in the EFE repo — the verifier source ships in
    the public GHCR files image (see :func:`fetch_ghcr_files`).
    """
    inv = json.loads(
        (ROOT / "data" / "efe" / "tasks" / "vlabench.json").read_text(encoding="utf-8")
    )
    paths: set[str] = set()
    for task in inv["tasks"]:
        for mode in task["modes"]:
            d = f"tasks/vlabench/{task['name']}-{mode}"
            # privileged keeps the instance at environment/; the standard
            # twin's frozen copy lives under its sim service (tests/ holds a
            # third copy for both).
            inst = (
                "environment/instance.json"
                if mode == "privileged"
                else "environment/sim/instance.json"
            )
            paths.add(f"{d}/{inst}")
            paths.add(f"{d}/instruction.md")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda p: gh_raw(
                    EFE_REPO,
                    "main",
                    p,
                    UPSTREAM_OUT / p.removeprefix("tasks/vlabench/"),
                ),
                sorted(paths),
            )
        )
    miss = [r for r in results if r.startswith("MISS")]
    print(
        f"[efe] vlabench tasks: {len(results) - len(miss)}/{len(results)}", flush=True
    )
    for m in miss[:10]:
        print("   ", m, flush=True)


def referenced_asset_paths() -> tuple[set[str], set[str]]:
    """Component/scene XML paths referenced by the mirrored instances."""
    components: set[str] = set()
    scenes: set[str] = set()
    for inst in sorted(UPSTREAM_OUT.glob("*-privileged/environment/instance.json")):
        data = json.loads(inst.read_text(encoding="utf-8"))
        task = data.get("episode", {}).get("task", {})
        for comp in task.get("components", []):
            if comp.get("xml_path"):
                components.add(comp["xml_path"])
        scene = task.get("scene", {})
        if scene.get("name"):
            scenes.add(scene["name"])
    return components, scenes


def fetch_with_assets(rel: str) -> list[str]:
    """Fetch one MJCF plus the mesh/texture files it references."""
    import posixpath

    dest = ASSETS_OUT / rel
    results = [hf(rel, dest)]
    if not (dest.is_file() and dest.stat().st_size > 0):
        return results
    base = posixpath.dirname(rel)
    for ref in xml_file_refs(dest):
        # upstream refs may carry "./" segments; normalize so neither the
        # URL nor the local path keeps the dot component.
        ref_rel = posixpath.normpath(posixpath.join(base, ref))
        results.append(hf(ref_rel, ASSETS_OUT / ref_rel))
    return results


def fetch_scene_assets(scene: str) -> list[str]:
    """Fetch a scene XML (name != filename) and everything it references."""
    listing = hf_list(f"scenes/{scene}")
    if not listing:
        print(f"[hf] LIST FAIL scenes/{scene}", flush=True)
        return [f"MISS scenes/{scene}"]
    results: list[str] = []
    xmls = [
        e["path"] for e in listing if e["type"] == "file" and e["path"].endswith(".xml")
    ]
    for xml in xmls:
        results.extend(fetch_with_assets(xml))
    return results


def fetch_panda() -> None:
    """Mirror the Panda robot MJCF (GitHub, MIT) with its mesh assets."""
    tree = gh_tree(VLB_REPO, VLB_REF, f"{PANDA_DIR}/assets")
    paths = [
        "panda.xml",
        "hand.xml",
        "panda_nohand.xml",
        "scene.xml",
        "LICENSE",
        "README.md",
    ]
    if tree:
        paths += [f"assets/{e['path']}" for e in tree if e["type"] == "blob"]
        grip = gh_tree(VLB_REPO, VLB_REF, f"{PANDA_DIR}/assets/panda_gripper")
        if grip:
            paths += [
                f"assets/panda_gripper/{e['path']}" for e in grip if e["type"] == "blob"
            ]
    else:
        print("[gh] panda assets listing failed; model files only", flush=True)
    out = ASSETS_OUT / "robots" / "franka_emika_panda"
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda p: gh_raw(VLB_REPO, VLB_REF, f"{PANDA_DIR}/{p}", out / p), paths
            )
        )
    miss = [r for r in results if r.startswith("MISS")]
    print(f"[gh] panda: {len(results) - len(miss)}/{len(results)}", flush=True)


def fetch_ghcr_files() -> None:
    """Pull the public GHCR ``efe:vlabench-files`` layer (verifier tooling).

    The task dirs' Dockerfiles COPY ``grade.py`` / ``vlb_tool.py`` /
    standard-service sources from this image layer; the EFE repo carries
    neither. GHCR anonymous token flow + single-layer tar, no docker needed.
    """
    import tarfile

    out = ROOT / "data" / "efe" / "upstream" / "images-vlabench"
    marker = out / "privileged" / "tests" / "grade.py"
    if marker.is_file() and marker.stat().st_size > 0:
        print("[ghcr] vlabench-files present", flush=True)
        return
    tag = "vlabench-files-0.5.2"
    token = json.loads(
        subprocess.run(
            [
                "curl",
                "-sfL",
                "--max-time",
                "60",
                "https://ghcr.io/token?scope=repository:mll-lab-nu/efe:pull",
            ],
            check=True,
            capture_output=True,
        ).stdout
    )["token"]
    manifest = json.loads(
        subprocess.run(
            [
                "curl",
                "-sfL",
                "--max-time",
                "60",
                "-H",
                f"Authorization: Bearer {token}",
                "-H",
                "Accept: application/vnd.oci.image.manifest.v1+json",
                f"https://ghcr.io/v2/mll-lab-nu/efe/manifests/{tag}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    )
    layer = manifest["layers"][0]["digest"]
    tar_path = ROOT / ".efe_stage" / f"efe-{tag}.tar"
    ok = get(f"https://ghcr.io/v2/mll-lab-nu/efe/blobs/{layer}", tar_path, tries=4)
    if not ok:
        print(f"[ghcr] layer fetch FAIL {layer}", flush=True)
        return
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            target = out / m.name
            if not target.resolve().is_relative_to(out.resolve()):
                continue
            if target.is_file() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            target.write_bytes(src.read())
            n += 1
    print(f"[ghcr] vlabench-files: +{n} files (layer {layer[:19]}…)", flush=True)


def write_provenance() -> None:
    """Record upstream refs and the mirrored file set next to the assets."""
    files = {}
    for f in sorted(ASSETS_OUT.rglob("*")):
        if f.is_file() and f.suffix != ".json":
            files[str(f.relative_to(ASSETS_OUT))] = f.stat().st_size
    (ASSETS_OUT / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "suite": "vlabench",
                "efe_repo": f"{EFE_REPO}@main",
                "vlabench_repo": f"{VLB_REPO}@{VLB_REF}",
                "hf_dataset": f"{HF_REPO}@{HF_REV}",
                "license_note": "VLABench assets: no redistribution licence; local mirror only.",
                "n_files": len(files),
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Round-robin the mirrors until present or the deadline hits."""
    while time.time() < DEADLINE:
        # GitHub mirrors via sparse clone first (partial clone fetches only
        # the wanted prefix; raw/API per-file fetches below fill gaps).
        # ``images/vlabench`` carries the verifier tooling (grade.py) that the
        # task dirs only reference from their Dockerfiles.
        print(
            "[efe]",
            gh_mirror_prefix(EFE_REPO, "main", "tasks/vlabench/", UPSTREAM_OUT),
            flush=True,
        )
        print(
            "[efe]",
            gh_mirror_prefix(
                EFE_REPO,
                "main",
                "images/vlabench/",
                ROOT / "data" / "efe" / "upstream" / "images-vlabench",
            ),
            flush=True,
        )
        print(
            "[gh]",
            gh_mirror_prefix(
                VLB_REPO,
                VLB_REF,
                f"{PANDA_DIR}/",
                ASSETS_OUT / "robots" / "franka_emika_panda",
            ),
            flush=True,
        )
        fetch_ghcr_files()
        fetch_efe_tasks()
        components, scenes = referenced_asset_paths()
        print(
            f"[plan] components: {len(components)} scenes: {sorted(scenes)}", flush=True
        )
        jobs: list[tuple[str, str]] = [("comp", c) for c in sorted(components)]
        jobs += [("scene", s) for s in sorted(scenes)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(
                pool.map(
                    lambda j: (
                        fetch_with_assets(j[1])
                        if j[0] == "comp"
                        else fetch_scene_assets(j[1])
                    ),
                    jobs,
                )
            )
        flat = [r for sub in results for r in sub]
        miss = [r for r in flat if r.startswith("MISS")]
        print(f"[hf] vlabench assets: {len(flat) - len(miss)}/{len(flat)}", flush=True)
        fetch_panda()
        if not miss:
            write_provenance()
            print("mirror complete", flush=True)
            return
        time.sleep(20)
    print("deadline reached", flush=True)


if __name__ == "__main__":
    main()
