"""Engine-level smoke for compiled VLABench tasks (slow).

Proves the first VLABench vertical slice composes on the real engine: the
franka MJCF robot loads, the frozen instance's MJCF component assets (static
furniture + freejoint-injected graspables) build with the declared poses, the
episode resets, and the configured success condition is False at the start
state (the EFE nop-check at Genesis level). One Genesis session per process
(``gs.init`` constraint), mirroring test_metaworldplus_smoke.py.

Skipped when the compiled configs or the upstream asset mirror are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAML = REPO_ROOT / "data" / "suites" / "efe_vlabench.yaml"
MODEL_MIRROR = REPO_ROOT / "assets" / "vlabench"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SUITE_YAML.is_file() or not MODEL_MIRROR.is_dir(),
        reason="compiled VLABench configs or vlabench mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    data = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    return [(SUITE_YAML.parent / p).resolve() for p in data["tasks"]]


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled VLABench task: build on Genesis, reset, nop must fail."""
    pytest.importorskip("genesis")
    # Registers franka_panda / efe_tabletop / efe_static.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.task_loader import load_task

    tasks = _task_paths()
    assert tasks, "suite lists no tasks"

    loaded = []
    for path in tasks:
        lt = load_task(path)
        loaded.append((path, lt))

    failures = []
    for path, lt in loaded:
        obs, _ = lt.reset_episode(seed=0)
        assert obs is not None, f"{path}: reset returned no observation"
        condition = lt.success_condition
        assert condition is not None, f"{path}: no success condition"
        if condition.evaluate(lt.env.scene, lt.env.robot) is not False:
            failures.append(path.name)
        entities = getattr(lt.env.scene, "entities", {})
        assert entities, f"{path}: scene has no task entities"
        dynamic = [
            name
            for name, spawn in ((s.name, s) for s in lt.spec.object_spawns)
            if not spawn.static
        ]
        assert dynamic, f"{path}: no dynamic graspables declared"
    assert not failures, (
        f"{len(failures)} task(s) score under the nop policy on the engine: "
        f"{failures}"
    )
