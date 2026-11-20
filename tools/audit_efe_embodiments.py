#!/usr/bin/env python3
"""Audit a requested EFE migration batch against the five robot embodiments.

The batch is curated in ``data/efe/embodiment_gate.yaml``: it names the EFE
suites to migrate and the tests each one contributes. This tool joins that
request with the three committed sources:

* ``data/efe/registry.json`` - the suite facts (paper test count, kinds);
* ``data/efe/robots.yaml`` - each robot's ``kinds`` (one or more of the five
  in-scope embodiments, or an out-of-scope kind such as ``quadruped``);
* ``data/efe/tasks/<suite>.json`` - the per-task robot assignment.

It derives a verdict per suite from the robots, not from the suite label:

* ``pass``    - every robot's kinds are inside the allowed five;
* ``partial`` - some robots are inside, others are outside;
* ``fail``    - no robot is inside the allowed five.

The curated ``decision`` must match the verdict (``migrate`` / ``partial`` /
``exclude``), and ``expected_tests`` must match the registry's paper count.
The report is written to ``data/efe/embodiment_gate.json``; ``--check`` is the
CI gate and fails on any structural error or stale artifact.

Usage::

    uv run python tools/audit_efe_embodiments.py
    uv run python tools/audit_efe_embodiments.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_EFE = REPO_ROOT / "data" / "efe"
REPORT = DATA_EFE / "embodiment_gate.json"

DECISION_BY_VERDICT = {"pass": "migrate", "partial": "partial", "fail": "exclude"}


def _load_yaml(path: Path) -> Any:
    """Parse one UTF-8 YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    """Parse one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _robot_kinds(robot: dict[str, Any]) -> tuple[str, ...]:
    """Return a robot's embodiment kinds (``kinds`` list or legacy ``kind``)."""
    raw = robot.get("kinds", robot.get("kind"))
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(kind) for kind in raw)


def _scope(robot_kinds: tuple[str, ...], allowed: tuple[str, ...]) -> bool:
    """True when every kind of the robot is inside the allowed set."""
    return bool(robot_kinds) and all(kind in allowed for kind in robot_kinds)


def _verdict(in_scope: int, total: int) -> str:
    """Classify a suite from its in-scope / total robot counts."""
    if total == 0:
        return "fail"
    if in_scope == total:
        return "pass"
    return "partial" if in_scope else "fail"


def _task_scope(
    data_dir: Path,
    suite: str,
    robots: dict[str, dict[str, Any]],
    allowed: tuple[str, ...],
) -> tuple[int, int, int, int, list[str], list[str], list[str]]:
    """Count and list in/out-of-scope tasks and test directories."""
    path = data_dir / "tasks" / f"{suite}.json"
    if not path.is_file():
        return 0, 0, 0, 0, [], [], [f"{suite}: task inventory missing: {path}"]
    inventory = _load_json(path)
    in_tasks = out_tasks = in_dirs = out_dirs = 0
    in_names: list[str] = []
    out_names: list[str] = []
    errors: list[str] = []
    for task in inventory.get("tasks", []):
        robot = robots.get(task.get("robot"))
        if robot is None:
            errors.append(
                f"{suite}/{task.get('name')}: robot {task.get('robot')!r} "
                "missing from robots.yaml"
            )
            continue
        kinds = _robot_kinds(robot)
        if not kinds:
            errors.append(f"{suite}/{task.get('name')}: robot has no kinds")
            continue
        modes = task.get("modes") or {}
        if _scope(kinds, allowed):
            in_tasks += 1
            in_dirs += len(modes)
            in_names.append(str(task.get("name")))
        else:
            out_tasks += 1
            out_dirs += len(modes)
            out_names.append(str(task.get("name")))
    return in_tasks, out_tasks, in_dirs, out_dirs, in_names, out_names, errors


def build_report(data_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Join the gate request with the registry and robot kinds."""
    gate = _load_yaml(data_dir / "embodiment_gate.yaml")
    registry = _load_json(data_dir / "registry.json")
    robots = {
        record["robot"]: record for record in _load_yaml(data_dir / "robots.yaml")
    }
    allowed_labels = gate["allowed_kinds"]
    allowed = tuple(allowed_labels)
    suites = {record["suite"]: record for record in registry["suites"]}

    errors: list[str] = []
    suite_reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for request in gate["batch"]["requested"]:
        name = request["suite"]
        if name in seen:
            errors.append(f"{name}: requested twice")
            continue
        seen.add(name)
        record = suites.get(name)
        if record is None:
            errors.append(f"{name}: not registered in registry.json")
            continue
        efe = record["efe"]
        if request["expected_tests"] != efe["tests"]:
            errors.append(
                f"{name}: expected_tests={request['expected_tests']} but "
                f"registry tests={efe['tests']}"
            )

        in_robots: list[str] = []
        out_robots: list[str] = []
        missing: list[str] = []
        robot_kinds: dict[str, list[str]] = {}
        for robot_name in efe["robots"]:
            robot = robots.get(robot_name)
            if robot is None:
                errors.append(f"{name}: robot {robot_name!r} missing from robots.yaml")
                missing.append(robot_name)
                continue
            kinds = _robot_kinds(robot)
            robot_kinds[robot_name] = list(kinds)
            if not kinds:
                errors.append(f"{name}: robot {robot_name!r} has no kinds")
                missing.append(robot_name)
            elif _scope(kinds, allowed):
                in_robots.append(robot_name)
            else:
                out_robots.append(robot_name)

        verdict = _verdict(len(in_robots), len(in_robots) + len(out_robots))
        expected_decision = DECISION_BY_VERDICT[verdict]
        if request["decision"] != expected_decision:
            errors.append(
                f"{name}: decision={request['decision']!r} but robot kinds "
                f"imply {expected_decision!r}"
            )

        suite_kinds = list(efe["kinds"])
        covered = {kind for kinds in robot_kinds.values() for kind in kinds}
        undeclared = [kind for kind in suite_kinds if kind not in covered]
        if undeclared:
            errors.append(
                f"{name}: suite kinds {undeclared} are not covered by any robot"
            )

        (
            in_tasks,
            out_tasks,
            in_dirs,
            out_dirs,
            in_names,
            out_names,
            task_errors,
        ) = _task_scope(data_dir, name, robots, allowed)
        errors.extend(task_errors)
        suite_reports.append(
            {
                "suite": name,
                "name": record["name"],
                "verdict": verdict,
                "decision": request["decision"],
                "note": request.get("note", ""),
                "tests": efe["tests"],
                "tasks": efe["tasks"],
                "suite_kinds": suite_kinds,
                "in_scope_kinds": sorted(k for k in suite_kinds if k in allowed),
                "out_of_scope_kinds": sorted(
                    k for k in suite_kinds if k not in allowed
                ),
                "in_scope_robots": in_robots,
                "out_of_scope_robots": out_robots,
                "missing_kinds": missing,
                "robot_kinds": robot_kinds,
                "in_scope_tasks": in_tasks,
                "out_of_scope_tasks": out_tasks,
                "in_scope_task_dirs": in_dirs,
                "out_of_scope_task_dirs": out_dirs,
                "in_scope_task_names": in_names,
                "out_of_scope_task_names": out_names,
            }
        )

    by_verdict = {"pass": 0, "partial": 0, "fail": 0}
    for report in suite_reports:
        by_verdict[report["verdict"]] += 1

    result = {
        "version": 1,
        "allowed_kinds": allowed_labels,
        "batch": {
            "id": gate["batch"]["id"],
            "note": gate["batch"].get("note", ""),
            "requested_suites": len(suite_reports),
            "tests": sum(report["tests"] for report in suite_reports),
            "tasks": sum(report["tasks"] for report in suite_reports),
            "in_scope_tasks": sum(report["in_scope_tasks"] for report in suite_reports),
            "out_of_scope_tasks": sum(
                report["out_of_scope_tasks"] for report in suite_reports
            ),
        },
        "totals": by_verdict,
        "suites": suite_reports,
        "errors": errors,
    }
    return result, errors


def render(report: dict[str, Any]) -> str:
    """Stable JSON rendering used by both writes and --check."""
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; ``--check`` diffs the committed artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed report is current (CI gate); does not write",
    )
    args = parser.parse_args(argv)

    report, errors = build_report(DATA_EFE)
    rendered = render(report)
    if not args.check:
        REPORT.write_text(rendered, encoding="utf-8")
    elif not REPORT.is_file():
        errors.append(f"report missing: {REPORT}")
    elif REPORT.read_text(encoding="utf-8") != rendered:
        errors.append(f"report out of sync: {REPORT} (run without --check)")

    totals = report["totals"]
    print(
        f"embodiment gate{' --check' if args.check else ' -> ' + str(REPORT)}"
        f" | suites={len(report['suites'])} pass={totals['pass']}"
        f" partial={totals['partial']} fail={totals['fail']}"
        f" | tasks in/out={report['batch']['in_scope_tasks']}"
        f"/{report['batch']['out_of_scope_tasks']}"
    )
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
