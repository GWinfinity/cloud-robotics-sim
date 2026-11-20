"""Tests for the embodiment gate (tools/audit_efe_embodiments.py).

The gate is the migration precondition: a suite may be migrated only after
its robots' kinds are allowed (v1: five embodiments; v2 adds ``quadruped``
under the documented 2026-11-20 user override that migrated all 13 MuJoCo
Playground tasks). These tests pin the committed batch and the verdicts
against the real data, offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit_efe_embodiments import (  # noqa: E402
    DATA_EFE,
    REPORT,
    build_report,
    main,
    render,
)

EXPECTED_TESTS = {
    "roboarc": 212,
    "embodied-olympics-fun": 179,
    "metaworldplus": 100,
    "embodied-safety-exam": 76,
    "dexhand": 60,
    "myosuite": 45,
    "bigym2": 39,
    "humanoidbench": 30,
    "mujoco-playground": 27,
}


def _report() -> dict:
    report, errors = build_report(DATA_EFE)
    assert errors == []
    return report


def test_gate_covers_the_requested_batch() -> None:
    """The nine requested suites and their paper test counts are pinned."""
    report = _report()
    suites = {item["suite"]: item for item in report["suites"]}
    assert set(suites) == set(EXPECTED_TESTS)
    assert {name: item["tests"] for name, item in suites.items()} == EXPECTED_TESTS
    assert report["totals"] == {"pass": 8, "partial": 0, "fail": 1}


def test_eight_suites_pass_the_gate() -> None:
    """Eight suites use only allowed kinds (incl. quadruped under v2)."""
    report = _report()
    passed = {item["suite"] for item in report["suites"] if item["verdict"] == "pass"}
    assert passed == {
        "roboarc",
        "embodied-olympics-fun",
        "metaworldplus",
        "embodied-safety-exam",
        "dexhand",
        "bigym2",
        "humanoidbench",
        "mujoco-playground",
    }
    allowed = set(report["allowed_kinds"])
    for item in report["suites"]:
        if item["verdict"] == "pass":
            for kinds in item["robot_kinds"].values():
                assert set(kinds) <= allowed


def test_myosuite_is_excluded_by_musculoskeletal_bodies() -> None:
    """MyoSuite's muscle-tendon models are outside the five embodiments."""
    item = next(s for s in _report()["suites"] if s["suite"] == "myosuite")
    assert item["verdict"] == "fail"
    assert item["decision"] == "exclude"
    assert item["in_scope_tasks"] == 0
    assert item["out_of_scope_tasks"] == 22
    assert item["out_of_scope_task_dirs"] == 45
    for kinds in item["robot_kinds"].values():
        assert "musculoskeletal" in kinds


def test_playground_passes_with_quadrupeds_under_the_override() -> None:
    """The 2026-11-20 override migrates all 13 Playground tasks."""
    item = next(s for s in _report()["suites"] if s["suite"] == "mujoco-playground")
    assert item["verdict"] == "pass"
    assert item["decision"] == "migrate"
    assert item["out_of_scope_kinds"] == []
    assert item["out_of_scope_robots"] == []
    assert item["in_scope_tasks"] == 13
    assert item["out_of_scope_tasks"] == 0
    assert item["in_scope_task_dirs"] == 27
    assert item["out_of_scope_task_dirs"] == 0


def test_committed_report_is_current(capsys) -> None:
    """The report on disk equals a fresh build and the CLI gate exits 0."""
    report = _report()
    assert REPORT.read_text(encoding="utf-8") == render(report)
    assert main(["--check"]) == 0
    assert "pass=8 partial=0 fail=1" in capsys.readouterr().out


def test_report_is_json_serialisable() -> None:
    """The committed artifact parses and carries no structural errors."""
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["errors"] == []
    assert data["batch"]["requested_suites"] == 9
    assert data["batch"]["in_scope_tasks"] == 373
    assert data["batch"]["out_of_scope_tasks"] == 22
