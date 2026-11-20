"""Offline nop gate for the compiled HumanoidBench EFE suite.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. HumanoidBench's rule is ``episode_return >= success_bar``; at the
frozen start the accumulated reward is 0 and every bar is >= 370, so the
task's own t=0 verdict (``info['hb'] = {"success": False, ...}``) and a
missing info must both read False. The engine smoke (test_hb_smoke.py)
proves the same on Genesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = "humanoidbench"


def _suite_configs() -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{SUITE}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


def test_compiled_hb_nop_fails() -> None:
    """Every compiled HumanoidBench task: verdict False before any step."""
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
        hb = task.get("hb") or {}
        assert hb.get("success_bar"), f"{path.name}: missing hb.success_bar"
        assert hb.get("task_short"), f"{path.name}: missing hb.task_short"
        condition = create_success_condition(success)
        # nop-safe: no info at all, and the task's own t=0 reset verdict.
        if condition.evaluate(None, None, None) is not False:
            failures.append(f"{path.name}: no-info verdict not False")
        reset_info = {"hb": {"success": False, "reward_sum": 0.0, "steps": 0}}
        if condition.evaluate(None, None, reset_info) is not False:
            failures.append(f"{path.name}: reset verdict not False")
    assert not failures, f"{SUITE}: nop gate failures: {failures}"
