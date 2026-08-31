"""Download a directory from a public GitHub repository via the Contents API.

Usage:
    python scripts/download_github_dir.py owner repo path outdir [--branch main]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


def api_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "lift-phase1-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(download_url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(download_url, headers={"User-Agent": "lift-phase1-downloader"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to download {download_url}: {last_error}")


def collect_files(owner: str, repo: str, path: str, branch: str = "main") -> List[dict]:
    """Recursively collect file metadata under a repo path."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    items: List[dict] = api_get(api_url)
    if not isinstance(items, list):
        raise RuntimeError(f"Expected list from {api_url}, got {type(items)}")
    files = []
    for item in items:
        if item["type"] == "dir":
            files.extend(collect_files(owner, repo, item["path"], branch))
        elif item["type"] == "file":
            files.append(item)
    return files


def download_dir(owner: str, repo: str, path: str, outdir: Path, branch: str = "main", workers: int = 2):
    files = collect_files(owner, repo, path, branch)
    print(f"Found {len(files)} files under {path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for item in files:
            rel = Path(item["path"]).relative_to(path)
            dest = outdir / rel
            fut = pool.submit(download_file, item["download_url"], dest)
            futures[fut] = item["path"]
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
                print(f"  OK {p}")
            except Exception as e:
                print(f"  FAIL {p}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download a GitHub repo directory")
    parser.add_argument("owner")
    parser.add_argument("repo")
    parser.add_argument("path")
    parser.add_argument("outdir")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.owner}/{args.repo}/{args.path} -> {outdir}")
    download_dir(args.owner, args.repo, args.path, outdir, args.branch, args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
