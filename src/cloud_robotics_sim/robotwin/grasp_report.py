"""Result records and report generation for the grasp-all-objects benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["FAIL_STAGES", "GraspRecord", "summarize", "write_report"]

#: Ordered failure stages (``success`` is the terminal success state).
FAIL_STAGES = (
    "load_fail",
    "plan_fail",
    "grasp_fail",
    "lift_fail",
    "transport_fail",
    "place_fail",
    "error",
)


@dataclass
class GraspRecord:
    """Outcome of one pick-and-place attempt on one object class."""

    class_name: str
    instance_index: int = -1
    kind: str = ""  # "glb" | "urdf"
    status: str = "error"  # "success" | one of FAIL_STAGES
    message: str = ""
    planner: str = ""  # "hierarchical_curobo" | "ompl"
    duration_s: float = 0.0
    final_pos: list[float] = field(default_factory=list)


def summarize(records: list[GraspRecord]) -> dict[str, Any]:
    """Aggregate records into a summary dict."""
    total = len(records)
    success = sum(1 for r in records if r.status == "success")
    by_stage: dict[str, int] = {s: 0 for s in FAIL_STAGES}
    for r in records:
        if r.status in by_stage:
            by_stage[r.status] += 1
    planners: dict[str, int] = {}
    for r in records:
        if r.planner:
            planners[r.planner] = planners.get(r.planner, 0) + 1
    return {
        "total": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
        "failures_by_stage": {k: v for k, v in by_stage.items() if v},
        "planner_usage": planners,
    }


def write_report(
    records: list[GraspRecord],
    out_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "extra": extra or {},
        "records": [asdict(r) for r in records],
    }
    json_path = out / "report.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Grasp-All-Objects Report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Total classes: **{summary['total']}**",
        f"- Success: **{summary['success']}** " f"({summary['success_rate']:.1%})",
        f"- Planner usage: {summary['planner_usage'] or 'n/a'}",
        "",
        "## Failures by stage",
        "",
    ]
    if summary["failures_by_stage"]:
        for stage, count in summary["failures_by_stage"].items():
            lines.append(f"- `{stage}`: {count}")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Per-class results",
        "",
        "| Class | Status | Planner | Duration (s) | Message |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        msg = r.message.replace("|", "\\|")[:80]
        lines.append(
            f"| {r.class_name} | {r.status} | {r.planner or '-'} "
            f"| {r.duration_s:.1f} | {msg} |"
        )
    md_path = out / "report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
