"""Unit tests for ``cloud_robotics_sim.robotwin.grasp_report``."""

from __future__ import annotations

import json
from pathlib import Path

from cloud_robotics_sim.robotwin.grasp_report import (
    GraspRecord,
    summarize,
    write_report,
)


def _records() -> list[GraspRecord]:
    return [
        GraspRecord(
            class_name="001_bottle", status="success", planner="ompl", duration_s=12.3
        ),
        GraspRecord(
            class_name="002_bowl",
            status="place_fail",
            planner="ompl",
            message="xy_err 0.12",
        ),
        GraspRecord(class_name="009_kettle", status="load_fail", message="no assets"),
        GraspRecord(
            class_name="010_pen", status="grasp_fail", planner="hierarchical_curobo"
        ),
    ]


def test_summarize() -> None:
    """Summary aggregates totals, stages and planner usage."""
    s = summarize(_records())
    assert s["total"] == 4
    assert s["success"] == 1
    assert s["success_rate"] == 0.25
    assert s["failures_by_stage"] == {"place_fail": 1, "load_fail": 1, "grasp_fail": 1}
    assert s["planner_usage"] == {"ompl": 2, "hierarchical_curobo": 1}


def test_summarize_empty() -> None:
    """Empty record list yields zeroed summary."""
    s = summarize([])
    assert s["total"] == 0
    assert s["success_rate"] == 0.0


def test_write_report(tmp_path: Path) -> None:
    """JSON + Markdown reports are written with expected content."""
    json_path, md_path = write_report(
        _records(), tmp_path, extra={"planner_mode": "auto"}
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 4
    assert payload["extra"]["planner_mode"] == "auto"
    assert len(payload["records"]) == 4
    md = md_path.read_text(encoding="utf-8")
    assert "001_bottle" in md
    assert "25.0%" in md
    assert "`load_fail`: 1" in md
