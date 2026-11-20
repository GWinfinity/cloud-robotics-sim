"""Offline nop gate for the compiled DexToolBench EFE suite.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. The engine smoke (test_dtb_smoke.py) proves the same on Genesis;
this gate checks every compiled config's ``dtb_all_goals`` verdict against
a stub world at the composed world poses. The verdict reads
``info['dtb_goals_reached']``, which the task zero-initialises — at the
frozen start no goal has been reached, so every verdict must read False.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
MIRROR = REPO_ROOT / "assets/dextoolbench"

pytestmark = pytest.mark.skipif(not MIRROR.is_dir(), reason="asset mirror not present")
SUITE = "dextoolbench"


def _suite_configs() -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{SUITE}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


class _StubWorld:
    """Poses from the extraction meta; contacts always empty."""

    def __init__(self, meta: dict, qpos: np.ndarray) -> None:
        self._meta = meta
        self._qpos = qpos

    def link_names(self) -> list[str]:
        return ["object", "table"]

    def geom_names(self) -> list[str]:
        return []

    def get_link_pos(self, name: str) -> np.ndarray:
        if name == "object":
            return np.asarray(self._meta["object_start_pos"], dtype=float)
        return np.zeros(3)

    def get_link_quat(self, name: str) -> np.ndarray:
        if name == "object":
            return np.asarray(self._meta["object_start_quat"], dtype=float)
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_qpos(self) -> np.ndarray:
        return self._qpos

    def get_self_contacts(self) -> dict:
        return {}

    def in_contact(self, _other: object) -> bool:
        return False


class _StubScene:
    def __init__(self, world: _StubWorld) -> None:
        self.entities = {"world": world}


def _load(config_path: Path) -> tuple[dict, _StubScene, dict]:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    success = data.get("success")
    assert isinstance(success, dict), f"{config_path.name}: missing success block"
    world_cfg = (data.get("task") or {}).get("world") or {}
    meta = json.loads((REPO_ROOT / world_cfg["meta_path"]).read_text(encoding="utf-8"))
    qpos = np.load(REPO_ROOT / world_cfg["qpos_path"])
    info = {"dtb_goals_reached": 0}
    return success, _StubScene(_StubWorld(meta, qpos)), info


def test_compiled_dtb_nop_fails() -> None:
    """Every compiled DexToolBench task: verdict False at the frozen start."""
    from cloud_robotics_sim.core.success import create_success_condition

    configs = _suite_configs()
    assert configs, f"{SUITE}: no compiled configs"
    failures = []
    for path in configs:
        success, scene, info = _load(path)
        condition = create_success_condition(success)
        if condition.evaluate(scene, robot=None, info=info) is not False:
            failures.append(path.name)
    assert not failures, (
        f"{SUITE}: {len(failures)} task(s) already succeed at the frozen start: "
        f"{failures}"
    )
