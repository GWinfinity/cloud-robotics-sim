#!/usr/bin/env python3
"""Build the Embodied First Exam (EFE) migration registry.

EFE (arXiv:2611.12177) is the umbrella benchmark this project ports into
Genesis: 35 suites / 47 robots / 1,156 tasks / 2,288 tests (456 of them core).
This tool is the index/audit half of that migration program, mirroring
``tools/build_embodiment_manifest.py``:

* Curated per-suite records live in ``data/efe/suites/*.yaml`` (paper facts:
  origin, backend, embodiment kinds, robots, task/test/core counts, plus the
  project's tier/status/target mapping).
* Machine inventory lives in ``data/efe/tasks/*.json`` (generated offline from
  an upstream snapshot by ``tools/fetch_efe_inventory.py``).
* Robot asset provenance lives in ``data/efe/robots.yaml``.

The build merges all three into ``data/efe/registry.json``; ``--check`` is the
CI gate and fails on any of:

* structural problems (unknown tier/status/kind, robot without a provenance
  record);
* drift between curated YAML counts and the upstream task inventory;
* drift of the in-paper totals away from the paper's Table 2
  (35 suites / 47 robots / 1,156 tasks / 2,288 tests / 456 core tasks).

Usage::

    uv run python tools/build_efe_registry.py           # write registry.json
    uv run python tools/build_efe_registry.py --check   # CI gate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/efe")

TIERS = ("T0", "T1", "T2", "T3", "TX")
STATUSES = ("ported", "partial", "planned", "blocked", "excluded")
KINDS = (
    "arm",
    "two_arms",
    "hand",
    "humanoid",
    "mobile_manip",
    "quadruped",
    "aerial",
    "car",
    "musculoskeletal",
)
ROBOT_SOURCES = (
    "have",
    "compose",
    "menagerie",
    "public",
    "efe_custom",
    "blocked",
    "out_of_paper",
)

#: Paper Table 1/2 totals for the in-paper suites (humanclaw is registered
#: but not counted by the paper).
PAPER = {
    "arxiv": "2611.12177",
    "repo": "https://github.com/mll-lab-nu/embodied-first-exam",
    "suites": 35,
    "robots": 47,
    "tasks": 1156,
    "tests": 2288,
    "core": 456,
}

REQUIRED_SUITE_KEYS = ("suite", "name", "efe", "project")
REQUIRED_EFE_KEYS = ("origin", "in_paper", "simulator", "kinds", "robots")


def load_yaml(path: Path) -> Any:
    """Parse one YAML file as UTF-8."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_suite_records(suites_dir: Path) -> list[dict[str, Any]]:
    """Load every curated suite record from a directory."""
    records = []
    for path in sorted(suites_dir.glob("*.yaml")):
        record = load_yaml(path)
        record["_path"] = path.name
        records.append(record)
    return records


def validate_structure(suite: dict[str, Any], robot_names: set[str]) -> list[str]:
    """Structural errors for one curated suite record (empty when valid)."""
    errors: list[str] = []
    label = suite.get("suite", "<unknown>")
    for key in REQUIRED_SUITE_KEYS:
        if key not in suite:
            errors.append(f"{label}: missing key {key!r}")
    efe = suite.get("efe") or {}
    project = suite.get("project") or {}
    for key in REQUIRED_EFE_KEYS:
        if key not in efe:
            errors.append(f"{label}: efe missing key {key!r}")
    for kind in efe.get("kinds") or []:
        if kind not in KINDS:
            errors.append(f"{label}: unknown kind {kind!r}")
    for robot in efe.get("robots") or []:
        if robot not in robot_names:
            errors.append(f"{label}: robot {robot!r} missing from robots.yaml")
    if project.get("tier") not in TIERS:
        errors.append(f"{label}: unknown tier {project.get('tier')!r}")
    if project.get("status") not in STATUSES:
        errors.append(f"{label}: unknown status {project.get('status')!r}")
    for key in ("target", "note"):
        if not project.get(key):
            errors.append(f"{label}: project missing {key!r}")
    return errors


def cross_check_inventory(
    suite: dict[str, Any], tasks_dir: Path
) -> tuple[list[str], list[str]]:
    """Drift between curated YAML facts and the upstream task inventory.

    Task and robot sets must match exactly. Test-dir counts may drift when
    the upstream repository evolves past the paper snapshot (e.g. RoboDojo
    gained modes after arXiv:2611.12177); that is a recorded warning, not a
    gate failure — the paper's Table 2 remains the canonical definition.
    """
    errors: list[str] = []
    warnings: list[str] = []
    label = suite["suite"]
    path = tasks_dir / f"{label}.json"
    if not path.is_file():
        return [f"{label}: no task inventory {path}"], warnings
    inv = json.loads(path.read_text(encoding="utf-8"))
    efe = suite["efe"]
    if inv["n_tasks"] != efe["tasks"]:
        errors.append(
            f"{label}: yaml tasks={efe['tasks']} != inventory {inv['n_tasks']}"
        )
    if inv["n_task_dirs"] != efe["tests"]:
        warnings.append(
            f"{label}: upstream drift — paper tests={efe['tests']}, "
            f"inventory dirs={inv['n_task_dirs']}"
        )
    inv_robots = set(inv["robots"])
    yaml_robots = set(efe["robots"])
    if inv_robots != yaml_robots:
        errors.append(
            f"{label}: yaml robots {sorted(yaml_robots)} != inventory {sorted(inv_robots)}"
        )
    return errors, warnings


def check_paper_totals(suites: list[dict[str, Any]]) -> list[str]:
    """In-paper totals must reproduce the paper's Table 2."""
    errors: list[str] = []
    in_paper = [s for s in suites if s["efe"].get("in_paper", True)]
    robot_universe = {r for s in in_paper for r in s["efe"]["robots"]}
    totals = {
        "suites": len(in_paper),
        "robots": len(robot_universe),
        "tasks": sum(s["efe"]["tasks"] for s in in_paper),
        "tests": sum(s["efe"]["tests"] for s in in_paper),
        "core": sum(s["efe"]["core"] or 0 for s in in_paper),
    }
    for key, expected in PAPER.items():
        if key in ("arxiv", "repo"):
            continue
        if totals[key] != expected:
            errors.append(f"paper totals drift: {key}={totals[key]} != {expected}")
    return errors


def build_registry(data_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Merge curated, inventory and robot records; validate all."""
    suites_dir = data_dir / "suites"
    tasks_dir = data_dir / "tasks"
    robots = load_yaml(data_dir / "robots.yaml")
    robot_names = {r["robot"] for r in robots}
    for robot in robots:
        if robot.get("source") not in ROBOT_SOURCES:
            raise ValueError(f"robots.yaml: {robot['robot']}: unknown source")

    suites = load_suite_records(suites_dir)
    attach_inventory(suites, tasks_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for suite in suites:
        suite_errors, suite_warnings = cross_check_inventory(suite, tasks_dir)
        errors.extend(validate_structure(suite, robot_names))
        errors.extend(suite_errors)
        warnings.extend(suite_warnings)
    errors.extend(check_paper_totals(suites))

    by_tier: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for suite in suites:
        project = suite["project"]
        by_tier[project["tier"]] = by_tier.get(project["tier"], 0) + 1
        by_status[project["status"]] = by_status.get(project["status"], 0) + 1

    registry = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER,
        "suites": [
            {
                "suite": s["suite"],
                "name": s["name"],
                "efe": s["efe"],
                "project": s["project"],
                "inventory": {
                    "n_tasks": s["_inventory"]["n_tasks"],
                    "n_task_dirs": s["_inventory"]["n_task_dirs"],
                },
            }
            for s in suites
        ],
        "robots": robots,
        "totals": {
            "suites": len(suites),
            "in_paper_suites": sum(1 for s in suites if s["efe"]["in_paper"]),
            "tasks": sum(s["efe"]["tasks"] for s in suites),
            "tests": sum(s["efe"]["tests"] for s in suites),
            "by_tier": by_tier,
            "by_status": by_status,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return registry, errors


def attach_inventory(suites: list[dict[str, Any]], tasks_dir: Path) -> None:
    """Best-effort attach of parsed inventory to suite records."""
    for suite in suites:
        path = tasks_dir / f"{suite['suite']}.json"
        if path.is_file():
            suite["_inventory"] = json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; --check is the CI gate."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on any validation error (CI gate); does not write",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    registry, errors = build_registry(data_dir)
    totals = registry["totals"]

    if not args.check:
        out = data_dir / "registry.json"
        out.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(
        f"registry{' --check' if args.check else ' -> ' + str(data_dir / 'registry.json')}"
        f" | suites={totals['suites']} (in-paper {totals['in_paper_suites']})"
        f" tasks={totals['tasks']} tests={totals['tests']}"
        f" tiers={totals['by_tier']} statuses={totals['by_status']}"
    )
    for warning in registry["warnings"]:
        logger.warning("%s", warning)
    for error in errors:
        logger.error("%s", error)
    if errors:
        logger.error("registry gate failed: %d error(s)", len(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
