"""Engine-level smoke for compiled LIBERO-MAX tasks (slow).

Proves the LIBERO-MAX port composes on the real engine: the franka MJCF
robot loads at LIBERO's mounted base pose, the probe-rebuilt scene (arena
table box + fixture/object MJCF at frozen world poses) builds, articulated
fixtures take their frozen joint states (open drawers), the episode resets
— applying any off-scene setup removals — and the configured success
condition is False at the start state (the EFE nop-check at Genesis level).
One Genesis session per process (``gs.init`` constraint), mirroring
test_vlabench_smoke.py.

Skipped when the compiled configs or the upstream asset mirror are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAML = REPO_ROOT / "data" / "suites" / "efe_libero-max.yaml"
MODEL_MIRROR = REPO_ROOT / "assets" / "liberomax"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SUITE_YAML.is_file() or not MODEL_MIRROR.is_dir(),
        reason="compiled LIBERO-MAX configs or liberomax mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    data = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    return [(SUITE_YAML.parent / p).resolve() for p in data["tasks"]]


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled LIBERO-MAX task: build on Genesis, reset, nop must fail."""
    pytest.importorskip("genesis")
    # Registers franka_panda / efe_tabletop / efe_liberomax.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.task_loader import load_task
    from cloud_robotics_sim.runtime.efe import EfeLiberoMaxTask

    tasks = _task_paths()
    assert tasks, "suite lists no tasks"

    loaded = []
    for path in tasks:
        loaded.append((path, load_task(path)))

    failures = []
    for path, lt in loaded:
        assert isinstance(lt.env.task.base_task, EfeLiberoMaxTask), (
            f"{path}: expected EfeLiberoMaxTask, got "
            f"{type(lt.env.task.base_task).__name__}"
        )
        obs, _ = lt.reset_episode(seed=0)
        assert obs is not None, f"{path}: reset returned no observation"
        condition = lt.success_condition
        assert condition is not None, f"{path}: no success condition"
        if condition.evaluate(lt.env.scene, lt.env.robot) is not False:
            failures.append(path.name)
        entities = getattr(lt.env.scene, "entities", {})
        assert entities, f"{path}: scene has no task entities"
    assert not failures, (
        f"{len(failures)} task(s) score under the nop policy on the engine: "
        f"{failures}"
    )
