"""Compiled EFE task configs must satisfy the port contract without Genesis.

For every task config referenced by a compiled EFE suite aggregate:

* it parses as a schema v1 task spec (core/task_loader);
* its success condition evaluates False on the task's initial object layout
  (the EFE nop-check: doing nothing must not score) — evaluated with stub
  entities at the poses the assets block declares, so no simulator runs.

Engine-level smoke (scene compose + step) lives in the slow embodiment-style
gates and is intentionally not duplicated here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from cloud_robotics_sim.core.task_loader import load_task_spec, sample_layout

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "data" / "suites"


def _suite_files() -> list[Path]:
    return sorted(SUITES_DIR.glob("efe_*.yaml"))


def _suite_task_paths(suite_file: Path) -> list[Path]:
    data = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    return [(suite_file.parent / p).resolve() for p in data["tasks"]]


def _stub_scene(spec):
    """Scene-like stub whose entities sit at the layout the spec samples.

    Stubs expose ``in_contact() -> False``: the offline nop-check assumes a
    contact-free start state unless a test overrides it (engine-level
    contact behaviour lives in the slow smoke gates).
    """
    layout = sample_layout(spec, seed=spec.evaluation.seeds[0])
    entities = {}
    for spawn in spec.object_spawns:
        pose = layout.get(spawn.name, {})
        entity = MagicMock()
        entity.get_pos.return_value = np.asarray(
            pose.get("position", spawn.position), dtype=float
        )
        quat = pose.get("orientation", spawn.orientation)
        entity.get_quat.return_value = np.asarray(quat, dtype=float)
        entity.in_contact = MagicMock(return_value=False)
        entities[spawn.name] = entity
    return SimpleNamespace(entities=entities)


@pytest.mark.parametrize("suite_file", _suite_files(), ids=lambda p: p.stem)
def test_compiled_suite_nop_must_fail(suite_file: Path) -> None:
    """Every compiled task: spec valid + success False under the nop policy."""
    task_paths = _suite_task_paths(suite_file)
    assert task_paths, f"{suite_file} references no tasks"
    robot = _stub_robot()
    for path in task_paths:
        spec = load_task_spec(path)
        scene = _stub_scene(spec)
        condition = spec.success_condition()
        assert condition is not None, f"{path}: no success condition"
        assert condition.evaluate(scene, robot) is False, (
            f"{path}: success already holds at the initial layout "
            f"(nop would score) — check the success block"
        )


def _stub_robot():
    """Robot stub whose end-effector link sits at the world origin.

    ee_near checks (MetaWorld reach family) must see a start pose away from
    every compiled target; the origin is outside all current goal windows.
    """

    class _Pose:
        pos = np.array([0.0, 0.0, 0.0])

    class _Entity:
        def get_link_pose(self, link):
            return _Pose()

    return SimpleNamespace(entity=_Entity())
