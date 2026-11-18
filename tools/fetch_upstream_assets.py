#!/usr/bin/env python3
"""Repo-local, idempotent fetch of upstream assets for the EFE P1 port.

Stages into the repository workspace (Windows temp cleanup eats /tmp):

* MetaWorld assets wheel/repo -> ``assets/metaworld/envs/assets/...``
  (robot MJCF + per-env XMLs + object meshes; MIT, provenance in README there)
* kindergarden (KinDER/TidyBot++) robot+env assets -> ``assets/kindergarden/``
* EFE P1 task files (instruction/instance/grader) -> ``data/efe/upstream/``

Every file is skipped once present and non-empty, so re-runs only fill gaps.
Designed for a flaky network: per-request retries with backoff, rounds until
the deadline, small responses only (non-recursive listings, one level down).
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("D:/githbi/genesis-cloud-sim")
DEADLINE = time.time() + 40 * 60

MW_REPO = "Farama-Foundation/Metaworld"
MW_REF = "59fc34d7768af9785e4688c3e1db671424f4a6c3"
KG_REPO = "Princeton-Robot-Planning-and-Learning/kindergarden"
KG_REF = "5b2dbac"
EFE_REPO = "mll-lab-nu/embodied-first-exam"
SUITES = ["metaworldplus", "libero-max", "vlabench", "robopaint", "kinder"]


def get(url: str, dest: Path | None = None, tries: int = 8) -> bool:
    """Fetch a URL to a path (or discard the body) with retry/backoff."""
    for attempt in range(tries):
        try:
            if dest is None:
                subprocess.run(
                    ["curl", "-sfL", "--max-time", "60", "-o", "NUL", url],
                    check=True,
                    capture_output=True,
                )
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                subprocess.run(
                    ["curl", "-sfL", "--max-time", "60", "-o", str(tmp), url],
                    check=True,
                    capture_output=True,
                )
                tmp.replace(dest)
            return True
        except subprocess.CalledProcessError:
            time.sleep(min(1.5**attempt, 15))
    return False


def api_tree(repo: str, ref: str, path: str) -> list[dict] | None:
    """List one git tree level via the GitHub API, with retries."""
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


def raw(repo: str, ref: str, path: str, dest: Path) -> str:
    """Fetch one upstream file into the repo-local mirror when missing."""
    if dest.is_file() and dest.stat().st_size > 0:
        return "have"
    ok = get(f"https://raw.githubusercontent.com/{repo}/{ref}/{path}", dest)
    return "ok" if ok else f"MISS {path}"


def fetch_metaworld() -> None:
    """Mirror the MetaWorld asset tree under assets/metaworld."""
    base = "metaworld/assets"
    out = ROOT / "assets" / "metaworld" / "envs" / "assets"
    paths: list[str] = []
    for sub in ("sawyer_xyz", "objects", "scene", "textures"):
        tree = api_tree(MW_REPO, MW_REF, f"{base}/{sub}")
        if not tree:
            print(f"[mw] LIST FAIL {sub}", flush=True)
            continue
        for e in tree:
            p = f"{sub}/{e['path']}"
            if e["type"] == "blob":
                paths.append(p)
            else:  # one more level (objects/assets, objects/meshes/...)
                sub2 = api_tree(MW_REPO, MW_REF, f"{base}/{p}")
                if sub2:
                    paths.extend(
                        f"{p}/{x['path']}" for x in sub2 if x["type"] == "blob"
                    )
                else:
                    print(f"[mw] LIST FAIL {p}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(lambda p: raw(MW_REPO, MW_REF, f"{base}/{p}", out / p), paths)
        )
    miss = [r for r in results if r.startswith("MISS")]
    print(f"[mw] {len(paths) - len(miss)}/{len(paths)} present", flush=True)
    for m in miss[:15]:
        print("   ", m, flush=True)


def resolve_kg_ref() -> str | None:
    """Resolve the short pinned kindergarden ref to a full SHA."""
    try:
        out = subprocess.run(
            [
                "curl",
                "-sfL",
                "--max-time",
                "60",
                f"https://api.github.com/repos/{KG_REPO}/commits/{KG_REF}",
            ],
            check=True,
            capture_output=True,
        )
        return json.loads(out.stdout)["sha"]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return None


def fetch_kindergarden() -> None:
    """Mirror kindergarden robot/env assets under assets/kindergarden."""
    ref = resolve_kg_ref()
    if not ref:
        print("[kg] cannot resolve ref", flush=True)
        return
    tree = api_tree(KG_REPO, ref, "")
    if not tree:
        print("[kg] root listing failed", flush=True)
        return
    dirs = [e["path"] for e in tree if e["type"] == "tree"]
    print("[kg] top dirs:", dirs, flush=True)
    keep = [
        d
        for d in dirs
        if any(
            k in d.lower()
            for k in (
                "asset",
                "robot",
                "tidybot",
                "env",
                "scene",
                "mimiclab",
                "urdfs",
                "models",
            )
        )
    ]
    out = ROOT / "assets" / "kindergarden"
    paths: list[str] = []
    for d in keep:
        t = api_tree(KG_REPO, ref, d)
        if not t:
            print(f"[kg] LIST FAIL {d}", flush=True)
            continue
        for e in t:
            p = f"{d}/{e['path']}"
            if e["type"] == "blob" and p.rsplit(".", 1)[-1].lower() in (
                "xml",
                "urdf",
                "stl",
                "obj",
                "json",
                "md",
                "py",
                "txt",
            ):
                paths.append(p)
            elif e["type"] == "tree":
                t2 = api_tree(KG_REPO, ref, p)
                if t2:
                    paths.extend(
                        f"{p}/{x['path']}"
                        for x in t2
                        if x["type"] == "blob"
                        and x["path"].rsplit(".", 1)[-1].lower()
                        in ("xml", "urdf", "stl", "obj", "json", "md", "py", "txt")
                    )
    print(f"[kg] candidate files: {len(paths)}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda p: raw(KG_REPO, ref, p, out / p), paths))
    miss = [r for r in results if r.startswith("MISS")]
    print(f"[kg] {len(paths) - len(miss)}/{len(paths)} present", flush=True)


def fetch_efe_tasks() -> None:
    """Mirror EFE P1 task dirs under data/efe/upstream/tasks."""
    out = ROOT / "data" / "efe" / "upstream" / "tasks"
    for suite in SUITES:
        inv = json.loads(
            (ROOT / "data" / "efe" / "tasks" / f"{suite}.json").read_text(
                encoding="utf-8"
            )
        )
        dirs = [f"{t['name']}-{m}" for t in inv["tasks"] for m in t["modes"]]
        tree = api_tree(EFE_REPO, "main", f"tasks/{suite}")
        test_files: set[str] = set()
        if tree:
            test_files = {
                f"tasks/{suite}/{e['path']}"
                for e in tree
                if e["type"] == "blob"
                and ("/tests/" in e["path"] or e["path"].endswith(("instruction.md",)))
            }
        else:
            print(f"[efe] LIST FAIL {suite}", flush=True)
            test_files = {
                f"tasks/{suite}/{d}/{name}"
                for d in dirs
                for name in ("instruction.md", "tests/grade.py")
            }
        for d in dirs:
            test_files.add(f"tasks/{suite}/{d}/environment/instance.json")
            test_files.add(f"tasks/{suite}/{d}/tests/grade.py")
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(
                pool.map(
                    lambda p: raw(EFE_REPO, "main", p, out / p.removeprefix("tasks/")),
                    sorted(test_files),
                )
            )
        miss = sum(1 for r in results if r.startswith("MISS"))
        print(f"[efe] {suite}: {len(results) - miss}/{len(results)}", flush=True)


def main() -> None:
    """Round-robin the mirrors until present or the deadline hits."""
    while time.time() < DEADLINE:
        fetch_metaworld()
        fetch_kindergarden()
        fetch_efe_tasks()
        done = (
            ROOT
            / "assets"
            / "metaworld"
            / "envs"
            / "assets"
            / "objects"
            / "xyz_base.xml"
        ).is_file()
        if done:
            print("core assets present; stopping early", flush=True)
            return
        time.sleep(20)
    print("deadline reached", flush=True)


if __name__ == "__main__":
    main()
