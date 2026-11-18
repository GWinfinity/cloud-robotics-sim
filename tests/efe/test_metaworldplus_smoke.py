"""Engine-level smoke for compiled EFE tasks (slow).

Proves the compiled MetaWorld+ configs compose on the real engine: sawyer MJCF
loads, the tabletop scene builds with the declared assets, the episode resets,
and the configured success condition is False at the start state (the EFE
nop-check at Genesis level). One Genesis session per process (``gs.init``
constraint), mirroring tests/backend/test_embodiment_build_gate.py.

Skipped when the compiled configs or the upstream model mirror are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_YAML = REPO_ROOT / "data" / "suites" / "efe_metaworldplus.yaml"
MODEL_MIRROR = REPO_ROOT / "assets" / "metaworld" / "envs" / "assets"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not SUITE_YAML.is_file() or not MODEL_MIRROR.is_dir(),
        reason="compiled EFE configs or metaworld mirror not present",
    ),
]


def _task_paths() -> list[Path]:
    import yaml

    data = yaml.safe_load(SUITE_YAML.read_text(encoding="utf-8"))
    return [(SUITE_YAML.parent / p).resolve() for p in data["tasks"]]


def test_compiled_tasks_compose_and_nop_fails() -> None:
    """Every compiled task: build on Genesis, reset, success False at start."""
    pytest.importorskip("genesis")
    # Registers sawyer / efe_tabletop / efe_static.
    import cloud_robotics_sim.runtime.main  # noqa: F401
    from cloud_robotics_sim.core.success import EeNearCondition
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
        if not isinstance(condition, EeNearCondition):
            assert entities, f"{path}: scene has no task entities"
    assert not failures, (
        f"{len(failures)} task(s) score under the nop policy on the engine "
        f"(settled start pose enters the goal window — fixture geometry "
        f"needed): {failures}"
    )
