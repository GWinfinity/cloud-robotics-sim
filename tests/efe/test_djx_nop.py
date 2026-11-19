"""Offline nop gate for the compiled DexJoCo EFE suite.

The ``djx_success`` verdict reads ``info['djx_success']``, which the task
False-initialises at reset — at the frozen start no family can already be
successful (EFE's invariant; the upstream evaluators were checked to fail
at the frozen state when the suite was mirrored). Contacts are stubbed
empty (nothing has been grasped/pressed yet), which is the correct nop
side for every contact-triggered family. The engine-level proof lives in
test_djx_smoke.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = "dexjoco"


def _suite_configs() -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{SUITE}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


def test_compiled_djx_nop_fails() -> None:
    """Every compiled DexJoCo task: verdict False at the frozen start."""
    from cloud_robotics_sim.core.success import create_success_condition

    configs = _suite_configs()
    assert configs, f"{SUITE}: no compiled configs"
    failures = []
    for path in configs:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        success = data.get("success")
        assert isinstance(success, dict), f"{path.name}: missing success block"
        condition = create_success_condition(success)
        scene = type("S", (), {"entities": {}})()
        if (
            condition.evaluate(scene, robot=None, info={"djx_success": False})
            is not False
        ):
            failures.append(path.name)
    assert not failures, (
        f"{SUITE}: {len(failures)} task(s) already succeed at the frozen start: "
        f"{failures}"
    )
