"""Engine-level smoke for the compiled HumanoidBench EFE suite (slow).

Proves the whole-world port composes on Genesis: each task's frozen scene
(Unitree H1 / H1+Shadow hands + furniture, one MJCF entity per world pack)
builds, the frozen qpos/qvel applies at reset, ``EfeHbTask`` runs its
reward/termination ports, and the compiled success verdict is False at the
start state (the EFE nop-check at engine level; every success bar is far
above the 0 return a frozen start accumulates). One Genesis session per
process (``gs.init`` constraint), mirroring test_pg_smoke.py.

Skipped when the compiled configs or the humanoidbench asset mirror are
absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAML = REPO_ROOT / "data" / "suites" / "efe_humanoidbench.yaml"
MODEL_MIRROR = REPO_ROOT / "assets" / "humanoidbench"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SUITE_YAML.is_file() or not MODEL_MIRROR.is_dir(),
        reason="compiled HumanoidBench configs or mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    data = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    return [(SUITE_YAML.parent / p).resolve() for p in data["tasks"]]


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled HumanoidBench task: build on Genesis, reset, nop fails."""
    pytest.importorskip("genesis")
    # Registers unitree_h1_embedded / efe_hb / hb_success.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.task_loader import load_task
    from cloud_robotics_sim.runtime.efe import EfeHbTask

    tasks = _task_paths()
    assert tasks, "suite lists no tasks"

    failures = []
    for path in tasks:
        # Load one at a time: each scene pins solver + mesh memory; without
        # teardown the accumulation starves later builds.
        lt = load_task(path)
        base = getattr(lt.env.task, "base_task", lt.env.task)
        assert isinstance(
            base, EfeHbTask
        ), f"{path}: expected EfeHbTask, got {type(base).__name__}"
        try:
            obs, info = lt.reset_episode(seed=0)
            assert obs is not None, f"{path}: reset returned no observation"
            condition = lt.success_condition
            assert condition is not None, f"{path}: no success condition"
            if condition.evaluate(lt.env.scene, lt.env.robot, info) is not False:
                failures.append(path.name)
            entities = getattr(lt.env.scene, "entities", {})
            assert "world" in entities, f"{path}: world entity missing"
        finally:
            destroy = getattr(lt.env.scene_backend, "destroy", None)
            if destroy is not None:
                destroy()
    assert not failures, (
        f"{len(failures)} task(s) score under the nop policy on the engine: "
        f"{failures}"
    )
