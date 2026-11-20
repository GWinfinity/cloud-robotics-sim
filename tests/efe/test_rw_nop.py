"""Offline nop gate for the compiled RoboWits EFE suite.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. The EFE verdict is the rcb_robowits settle window (hold, then
the goal at every check step with the graded objects at rest); at the
frozen start the family's predicate does not hold (dominos: the red
block is not on the target; stand_pages: the pages lie flat; stack_cubes:
no stack), so the task's own t=0 verdict and a missing info must both
read False. The engine smoke (test_rw_smoke.py) proves the same on
Genesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = "robowits"


def _suite_configs() -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{SUITE}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


def test_compiled_rw_nop_fails() -> None:
    """Every compiled RoboWits task: verdict False before any step."""
    from cloud_robotics_sim.core.success import create_success_condition

    configs = _suite_configs()
    assert configs, f"{SUITE}: no compiled configs"
    failures = []
    for path in configs:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        success = data.get("success")
        assert isinstance(success, dict), f"{path.name}: missing success block"
        task = data.get("task") or {}
        rw = task.get("rw") or {}
        assert rw.get("family"), f"{path.name}: missing rw.family"
        assert rw.get("settle"), f"{path.name}: missing rw.settle"
        condition = create_success_condition(success)
        if condition.evaluate(None, None, None) is not False:
            failures.append(f"{path.name}: no-info verdict not False")
        reset_info = {"rw": {"success": False, "goal": False, "hold_steps": 0}}
        if condition.evaluate(None, None, reset_info) is not False:
            failures.append(f"{path.name}: reset verdict not False")
    assert not failures, f"{SUITE}: nop gate failures: {failures}"
