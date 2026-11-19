"""Engine-level smoke for the compiled DexJoCo EFE suite (slow).

Proves the whole-world port composes on Genesis: each task's frozen scene
(two I2RT YAM arms + table + objects, one MJCF entity with free-jointed
graspables and finger equality couplings) builds, the frozen qpos/qvel
applies at reset, ``EfeDjxTask`` runs its runtime hooks, and the compiled
success verdict is False at the start state (the EFE nop-check at engine
level). One Genesis session per process (``gs.init`` constraint), mirroring
test_robocasa_smoke.py.

Skipped when the compiled configs or the dexjoco asset mirror are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAML = REPO_ROOT / "data" / "suites" / "efe_dexjoco.yaml"
MODEL_MIRROR = REPO_ROOT / "assets" / "dexjoco"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SUITE_YAML.is_file() or not MODEL_MIRROR.is_dir(),
        reason="compiled ABC configs or dexjoco mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    data = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    return [(SUITE_YAML.parent / p).resolve() for p in data["tasks"]]


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled ABC task: build on Genesis, reset, nop must fail."""
    pytest.importorskip("genesis")
    # Registers kuka_iiwa_sharpa / efe_world / efe_djx.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.task_loader import load_task
    from cloud_robotics_sim.runtime.efe import EfeDjxTask

    tasks = _task_paths()
    assert tasks, "suite lists no tasks"

    failures = []
    for path in tasks:
        # Load one at a time: each scene pins solver + mesh memory; without
        # teardown the accumulation starves later builds.
        lt = load_task(path)
        base = getattr(lt.env.task, "base_task", lt.env.task)
        assert isinstance(
            base, EfeDjxTask
        ), f"{path}: expected EfeDjxTask, got {type(base).__name__}"
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
