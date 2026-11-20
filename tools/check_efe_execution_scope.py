#!/usr/bin/env python3
"""Audit the executable scope after the EFE embodiment decision.

``data/efe/exclusions.yaml`` is the human-curated decision record: MyoSuite is
excluded as a whole and MuJoCo Playground's nine quadruped tasks are excluded
individually. This tool joins that decision with
``data/efe/embodiment_gate.json`` and fails when:

* the exclusion manifest drifts from the gate's out-of-scope task/robot/kinds;
* the policy is changed to relax the five-embodiment gate;
* an excluded task leaks into a task config, a suite aggregate, or the
  executable ``tasks`` / ``pending_tasks`` half of a compile knowledge base;
* an excluded task is copied into the compile knowledge base.

The paper registry in ``data/efe/suites/`` deliberately keeps MyoSuite and all
Playground tasks so the paper totals remain reconcilable; only executable
artifacts are stripped.

Usage::

    uv run python tools/check_efe_execution_scope.py
    uv run python tools/check_efe_execution_scope.py --check
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
EXCLUSIONS = DATA_EFE / "exclusions.yaml"
GATE_REPORT = DATA_EFE / "embodiment_gate.json"
CONFIGS_EFE = REPO_ROOT / "configs" / "tasks" / "efe"
SUITES_OUT = REPO_ROOT / "data" / "suites"


def _load_yaml(path: Path) -> Any:
    """Parse one UTF-8 YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    """Parse one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_exclusions(path: Path = EXCLUSIONS) -> dict[str, Any]:
    """Load and lightly validate the exclusion manifest."""
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    if data.get("version") != 1:
        raise ValueError(f"{path}: unsupported version {data.get('version')!r}")
    return data


def _manifest_by_suite(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["suite"]: record for record in manifest.get("suites") or []}


def _gate_by_suite(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["suite"]: record for record in report.get("suites") or []}


def _check_manifest_against_gate(
    manifest: dict[str, Any], gate: dict[str, Any]
) -> list[str]:
    """The decision record and the gate must describe the exact same scope."""
    errors: list[str] = []
    policy = manifest.get("policy") or {}
    if policy.get("decision") != "exclude":
        errors.append(
            "policy.decision must be 'exclude' (excluded suites stay excluded)"
        )
    if policy.get("gate_relaxed") is False:
        pass  # the strict five-kind gate: nothing to document
    elif policy.get("gate_relaxed") is True:
        note = str(policy.get("note") or "")
        if "覆盖" not in note and "override" not in note.lower():
            errors.append(
                "policy.gate_relaxed=true requires policy.note to record the "
                "user override that relaxed the gate"
            )
    else:
        errors.append("policy.gate_relaxed must be a boolean")

    manifest_suites = _manifest_by_suite(manifest)
    gate_suites = _gate_by_suite(gate)
    expected = {
        name: record
        for name, record in gate_suites.items()
        if record.get("out_of_scope_task_names")
    }
    missing = sorted(set(expected) - set(manifest_suites))
    extra = sorted(set(manifest_suites) - set(expected))
    for name in missing:
        errors.append(f"{name}: gate excludes tasks but manifest has no entry")
    for name in extra:
        errors.append(f"{name}: manifest entry is not excluded by the gate")

    for name in sorted(set(expected) & set(manifest_suites)):
        record = manifest_suites[name]
        gate_record = expected[name]
        tasks = record.get("tasks") or []
        if len(tasks) != len(set(tasks)):
            errors.append(f"{name}: duplicate task in exclusion manifest")
        if set(tasks) != set(gate_record["out_of_scope_task_names"]):
            only_manifest = sorted(
                set(tasks) - set(gate_record["out_of_scope_task_names"])
            )
            only_gate = sorted(set(gate_record["out_of_scope_task_names"]) - set(tasks))
            errors.append(
                f"{name}: exclusion task drift "
                f"(manifest-only={only_manifest}, gate-only={only_gate})"
            )
        if set(record.get("robots") or []) != set(
            gate_record.get("out_of_scope_robots") or []
        ):
            errors.append(f"{name}: exclusion robot set differs from gate")
        if set(record.get("kinds") or []) != set(
            gate_record.get("out_of_scope_kinds") or []
        ):
            errors.append(f"{name}: exclusion kinds differ from gate")
        expected_scope = "suite" if gate_record.get("in_scope_tasks") == 0 else "tasks"
        if record.get("scope") != expected_scope:
            errors.append(
                f"{name}: scope={record.get('scope')!r}, expected {expected_scope!r}"
            )
    return errors


def _task_names_in_config(path: Path) -> set[str]:
    """Return the task's declared name and filename stem."""
    names = {path.stem}
    try:
        data = _load_yaml(path) or {}
    except yaml.YAMLError:
        return names
    task = data.get("task") if isinstance(data, dict) else None
    if isinstance(task, dict) and isinstance(task.get("name"), str):
        names.add(task["name"])
    return names


def _check_executable_leaks(
    manifest: dict[str, Any],
    configs_root: Path,
    suites_root: Path,
    compile_root: Path,
) -> list[str]:
    """Excluded names may not appear in executable configs or pending tasks."""
    errors: list[str] = []
    for record in manifest.get("suites") or []:
        suite = record["suite"]
        excluded = set(record.get("tasks") or [])

        suite_dir = configs_root / suite
        if suite_dir.is_dir():
            for path in sorted(suite_dir.glob("*.yaml")):
                leaked = sorted(_task_names_in_config(path) & excluded)
                for name in leaked:
                    errors.append(f"{suite}: excluded task config present: {path}")

        aggregate = suites_root / f"efe_{suite}.yaml"
        if aggregate.is_file():
            data = _load_yaml(aggregate) or {}
            for reference in data.get("tasks") or []:
                target = (aggregate.parent / reference).resolve()
                if target.stem in excluded:
                    errors.append(
                        f"{suite}: aggregate references excluded task: {reference}"
                    )
                    continue
                if target.is_file():
                    leaked = sorted(_task_names_in_config(target) & excluded)
                    for name in leaked:
                        errors.append(
                            f"{suite}: aggregate task {name} is excluded: {target}"
                        )

        kb_path = compile_root / f"{suite}.yaml"
        if not kb_path.is_file():
            continue
        kb = _load_yaml(kb_path) or {}
        executable = set((kb.get("tasks") or {}).keys())
        executable |= set(kb.get("pending_tasks") or [])
        leaked = sorted(executable & excluded)
        for name in leaked:
            errors.append(
                f"{suite}: excluded task in executable KB tasks/pending_tasks: {name}"
            )
        if "excluded_tasks" in kb or "excluded_reasons" in kb:
            errors.append(
                f"{suite}: excluded tasks must stay only in "
                "data/efe/exclusions.yaml, not be copied into the compile KB"
            )
    return errors


def audit_scope(
    manifest: dict[str, Any],
    gate: dict[str, Any],
    configs_root: Path = CONFIGS_EFE,
    suites_root: Path = SUITES_OUT,
    compile_root: Path = DATA_EFE / "compile",
) -> tuple[list[str], dict[str, Any]]:
    """Run both consistency checks and return errors plus a summary."""
    errors = _check_manifest_against_gate(manifest, gate)
    errors.extend(
        _check_executable_leaks(manifest, configs_root, suites_root, compile_root)
    )
    suites = manifest.get("suites") or []
    summary = {
        "version": 1,
        "policy": manifest.get("policy"),
        "excluded_suites": [record["suite"] for record in suites],
        "excluded_tasks": sum(len(record.get("tasks") or []) for record in suites),
        "errors": errors,
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; ``--check`` is accepted as the CI-gate spelling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusions", default=str(EXCLUSIONS))
    parser.add_argument("--gate", default=str(GATE_REPORT))
    parser.add_argument("--configs-root", default=str(CONFIGS_EFE))
    parser.add_argument("--suites-root", default=str(SUITES_OUT))
    parser.add_argument("--compile-root", default=str(DATA_EFE / "compile"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI-gate spelling (the tool is read-only either way)",
    )
    args = parser.parse_args(argv)

    manifest = load_exclusions(Path(args.exclusions))
    gate = _load_json(Path(args.gate))
    errors, summary = audit_scope(
        manifest,
        gate,
        configs_root=Path(args.configs_root),
        suites_root=Path(args.suites_root),
        compile_root=Path(args.compile_root),
    )
    print(
        f"execution scope{' --check' if args.check else ''}"
        f" | suites={len(summary['excluded_suites'])}"
        f" tasks={summary['excluded_tasks']} errors={len(errors)}"
    )
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
