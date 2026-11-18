#!/usr/bin/env python3
"""Fetch the Embodied First Exam (EFE) task inventory into this repository.

EFE (arXiv:2611.12177, https://github.com/mll-lab-nu/embodied-first-exam)
packages every task as ``tasks/<suite>/<task>-<mode>/task.toml`` inside its
Harbor-format repository. This tool reads a *local snapshot* of that
repository (a sparse, blob-filtered checkout is enough — only ``task.toml``
files are needed) and emits one ``data/efe/tasks/<suite>.json`` per suite with
the per-task facts the migration registry cross-checks against.

The tool never touches the network: fetching the snapshot is a manual step,
documented in ``data/efe/README.md``. The emitted JSON files are the committed
artifacts; CI and ``tools/build_efe_registry.py`` consume them offline.

Usage::

    uv run python tools/fetch_efe_inventory.py --snapshot /path/to/efe-snapshot
    uv run python tools/fetch_efe_inventory.py --snapshot ... --out data/efe/tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MODES = ("privileged", "standard", "realtime")

SOURCE_REPO = "https://github.com/mll-lab-nu/embodied-first-exam"


def parse_task_toml(path: Path) -> dict:
    """Return the metadata fields the inventory keeps, from one task.toml."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    agent = data.get("agent", {})
    verifier = data.get("verifier", {})
    return {
        "family": meta.get("family"),
        "robot": meta.get("robot"),
        "simulator": meta.get("simulator"),
        "dynamics": meta.get("dynamics"),
        "task_type": meta.get("task_type"),
        "instance_seed": meta.get("instance_seed"),
        "oracle": meta.get("oracle"),
        "protocol": meta.get("protocol"),
        "agent_timeout_sec": agent.get("timeout_sec"),
        "verifier_timeout_sec": verifier.get("timeout_sec"),
    }


def split_mode(task_dir: str) -> tuple[str, str] | None:
    """Split ``<name>-<mode>`` into (name, mode); None when the suffix is odd."""
    stem, sep, suffix = task_dir.rpartition("-")
    if not sep or suffix not in MODES:
        logger.warning("task dir %r has no recognized mode suffix; skipped", task_dir)
        return None
    return stem, suffix


def collect_suite(snapshot: Path, suite_dir: Path) -> dict:
    """Build the inventory record for one suite directory."""
    grouped: dict[str, dict] = {}
    for toml_path in sorted(suite_dir.glob("*/task.toml")):
        parsed = split_mode(toml_path.parent.name)
        if parsed is None:
            continue
        name, mode = parsed
        facts = parse_task_toml(toml_path)
        entry = grouped.setdefault(
            name,
            {
                "name": name,
                "family": facts["family"],
                "robot": facts["robot"],
                "simulator": facts["simulator"],
                "dynamics": facts["dynamics"],
                "task_type": facts["task_type"],
                "instance_seed": facts["instance_seed"],
                "modes": {},
            },
        )
        entry["modes"][mode] = {
            "oracle": facts["oracle"],
            "protocol": facts["protocol"],
            "agent_timeout_sec": facts["agent_timeout_sec"],
            "verifier_timeout_sec": facts["verifier_timeout_sec"],
        }
    return {
        "suite": suite_dir.name,
        "n_tasks": len(grouped),
        "n_task_dirs": sum(len(e["modes"]) for e in grouped.values()),
        "robots": sorted({e["robot"] for e in grouped.values() if e["robot"]}),
        "simulators": sorted(
            {e["simulator"] for e in grouped.values() if e["simulator"]}
        ),
        "tasks": sorted(grouped.values(), key=lambda e: e["name"]),
    }


def build_inventory(snapshot: Path, out: Path) -> list[dict]:
    """Parse all task.toml files and write per-suite JSON files."""
    tasks_root = snapshot / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"snapshot {snapshot} has no tasks/ directory")
    out.mkdir(parents=True, exist_ok=True)
    suites = []
    for suite_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        record = collect_suite(snapshot, suite_dir)
        record["source"] = {
            "repo": SOURCE_REPO,
            "ref": "main",
            "snapshot_taken": datetime.now(timezone.utc).isoformat(),
            "note": "sparse blob-filtered checkout; task.toml files only",
        }
        (out / f"{suite_dir.name}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        suites.append(record)
        logger.info(
            "%-28s tasks=%4d dirs=%4d robots=%s",
            record["suite"],
            record["n_tasks"],
            record["n_task_dirs"],
            ",".join(record["robots"]),
        )
    return suites


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, required=True, help="local EFE repo snapshot"
    )
    parser.add_argument("--out", type=Path, default=Path("data/efe/tasks"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    suites = build_inventory(args.snapshot, args.out)
    n_tasks = sum(s["n_tasks"] for s in suites)
    n_dirs = sum(s["n_task_dirs"] for s in suites)
    print(f"suites={len(suites)} tasks={n_tasks} task_dirs={n_dirs} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
