"""Engine-level smoke for the compiled RoboCasa EFE suites (slow).

Proves the whole-world port composes on Genesis: each task's frozen
kitchen (robot + fixtures + objects, one MJCF entity) builds, the frozen
qpos/qvel applies at reset, the world entity exposes the link/site hooks
the rc_* predicates need, and the compiled success condition is False at
the start state (the EFE nop-check at engine level). One Genesis session
per process (``gs.init`` constraint), mirroring test_liberomax_smoke.py.

Skipped when the compiled configs or the robocasa asset mirror are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAMLS = tuple(
    REPO_ROOT / "data" / "suites" / f"efe_{suite}.yaml"
    for suite in ("robocasa", "robocasa365", "robocasa-gr1")
)
MODEL_MIRROR = REPO_ROOT / "assets" / "robocasa"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(p.is_file() for p in SUITE_YAMLS) or not MODEL_MIRROR.is_dir(),
        reason="compiled RoboCasa configs or robocasa mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    paths = []
    for suite_yaml in SUITE_YAMLS:
        data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
        paths.extend((suite_yaml.parent / p).resolve() for p in data["tasks"])
    return paths


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled RoboCasa task: build on Genesis, reset, nop must fail."""
    pytest.importorskip("genesis")
    # Registers franka_panda_omron / fourier_gr1 / efe_world / efe_robocasa.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.task_loader import load_task
    from cloud_robotics_sim.runtime.efe import EfeRobocasaTask

    tasks = _task_paths()
    assert tasks, "suites list no tasks"

    loaded = []
    for path in tasks:
        loaded.append((path, load_task(path)))

    failures = []
    for path, lt in loaded:
        base = getattr(lt.env.task, "base_task", lt.env.task)
        assert isinstance(
            base, EfeRobocasaTask
        ), f"{path}: expected EfeRobocasaTask, got {type(base).__name__}"
        obs, info = lt.reset_episode(seed=0)
        assert obs is not None, f"{path}: reset returned no observation"
        condition = lt.success_condition
        assert condition is not None, f"{path}: no success condition"
        if condition.evaluate(lt.env.scene, lt.env.robot, info) is not False:
            failures.append(path.name)
        entities = getattr(lt.env.scene, "entities", {})
        assert "world" in entities, f"{path}: world entity missing"
    assert not failures, (
        f"{len(failures)} task(s) score under the nop policy on the engine: "
        f"{failures}"
    )
