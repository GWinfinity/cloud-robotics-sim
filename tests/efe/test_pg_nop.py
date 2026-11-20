"""Offline nop gate for the compiled MuJoCo Playground EFE suite.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. The engine smoke (test_pg_smoke.py) proves the same on Genesis;
this gate checks every compiled config's ``pg_success`` verdict, which
forwards ``info['pg']['success']`` — missing info and the task's own reset
info (``{"success": False, "steps": 0}``) must both read False.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = "mujoco-playground"


def _suite_configs() -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{SUITE}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


def test_compiled_pg_nop_fails() -> None:
    """Every compiled Playground task: verdict False before any step."""
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
        pg = task.get("pg") or {}
        assert pg.get("task_kind"), f"{path.name}: missing pg.task_kind"
        condition = create_success_condition(success)
        # nop-safe: no info at all, and the task's own t=0 reset verdict.
        if condition.evaluate(None, None, None) is not False:
            failures.append(f"{path.name}: no-info verdict not False")
        reset_info = {"pg": {"success": False, "steps": 0}}
        if condition.evaluate(None, None, reset_info) is not False:
            failures.append(f"{path.name}: reset verdict not False")
    assert not failures, f"{SUITE}: nop gate failures: {failures}"
