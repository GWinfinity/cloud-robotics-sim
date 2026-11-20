"""Offline nop gate for the compiled ABC Sim EFE suite.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. The engine smoke (test_abc_smoke.py) proves the same on Genesis;
this gate checks every compiled config's success spec against a stub world
rebuilt from the extraction meta (compiled world poses + frozen qpos) —
no engine, so it runs in CI without the asset mirror.

The ABC success blocks are the EFE-level verdicts (``abc_hold`` /
``abc_ball_tray`` with countdown / ``abc_drawer_sequence``), all fed by
``EfeAbcTask``'s per-step info counters; at the frozen start the counters
are at zero and the ball-tray countdown has not run, so every verdict must
read False. Contacts are stubbed empty (the correct nop side: nothing has
been grasped yet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
MIRROR = REPO_ROOT / "assets/abc-sim"

pytestmark = pytest.mark.skipif(not MIRROR.is_dir(), reason="asset mirror not present")
SUITE = "abc-sim"


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
        return list(self._meta["bodies"])

    def geom_names(self) -> list[str]:
        return []

    def get_link_pos(self, name: str) -> np.ndarray:
        return np.asarray(self._meta["bodies"][name]["world"], dtype=float)

    def get_link_quat(self, name: str) -> np.ndarray:
        return np.asarray(self._meta["bodies"][name]["world_quat"], dtype=float)

    def get_joint_position(self, name: str) -> float:
        for record in self._meta["joints"]:
            if record.get("name") == name:
                return float(self._qpos[int(record["qposadr"])])
        raise KeyError(name)

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
    task_cfg = data.get("task") or {}
    success = data.get("success")
    assert isinstance(success, dict), f"{config_path.name}: missing success block"
    world_cfg = task_cfg.get("world") or {}
    meta_rel = world_cfg.get("meta_path")
    qpos_rel = world_cfg.get("qpos_path")
    assert meta_rel and qpos_rel, f"{config_path.name}: missing world block"
    meta = json.loads((REPO_ROOT / meta_rel).read_text(encoding="utf-8"))
    qpos = np.load(REPO_ROOT / qpos_rel)
    scene = _StubScene(_StubWorld(meta, qpos))
    info = {
        "abc_success_run": 0,
        "abc_tray_complete": False,
        "abc_drawer_completed": 0,
        "abc_sweep_active": [False]
        * len(((task_cfg.get("abc") or {}).get("sweep") or {}).get("scraps", [])),
    }
    return success, scene, info


def test_compiled_abc_nop_fails() -> None:
    """Every compiled ABC task: success verdict must be False at the frozen start."""
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
