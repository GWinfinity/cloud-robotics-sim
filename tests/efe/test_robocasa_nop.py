"""Offline nop gate for the compiled RoboCasa EFE suites.

EFE's verifier invariant: the frozen start state must NOT already satisfy
the task. The engine smoke (test_robocasa_smoke.py) proves the same on
Genesis; this gate checks every compiled config's success spec against a
stub world rebuilt from the extraction meta (compiled world poses +
frozen qpos) — no engine, so it runs in CI without the asset mirror.

Contacts are stubbed empty: predicates that need contact (rc_contact,
rc_in_receptacle, rc_contact_any) therefore evaluate False, which is the
correct nop side for every placement task. Joint/window predicates read
the real frozen qpos, so a task whose start state already passes a joint
check (e.g. a ``close`` task on an already-closed drawer) would surface
here as a binding bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]

SUITES = ("robocasa", "robocasa365", "robocasa-gr1")


def _suite_configs(suite: str) -> list[Path]:
    suite_yaml = REPO_ROOT / "data" / "suites" / f"efe_{suite}.yaml"
    if not suite_yaml.is_file():
        return []
    import yaml

    data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    return [(suite_yaml.parent / rel).resolve() for rel in data["tasks"]]


def _joint_q_offsets(meta: dict) -> dict[str, tuple[int, int]]:
    """(start, width) of each joint's qpos segment, in document order."""
    sizes = {"free": 7, "ball": 4}
    out: dict[str, tuple[int, int]] = {}
    cursor = 0
    for record in meta["joints"]:
        width = sizes.get(record["type"], 1)
        out[record["name"]] = (cursor, width)
        cursor += width
    return out


class _StubWorld:
    """Poses from the extraction meta; contacts always empty."""

    def __init__(self, meta: dict, qpos: np.ndarray) -> None:
        self._meta = meta
        self._qpos = qpos
        self._offsets = _joint_q_offsets(meta)
        self._rc_sites = {
            name: (
                rec["body"],
                tuple(float(v) for v in str(rec["pos"]).split()),
            )
            for name, rec in (meta.get("sites") or {}).items()
        }

    def link_names(self) -> list[str]:
        return list(self._meta["bodies"])

    def geom_names(self) -> list[str]:
        return []

    def get_link_pos(self, name: str) -> np.ndarray:
        return np.asarray(self._meta["bodies"][name]["world"], dtype=float)

    def get_link_quat(self, name: str) -> np.ndarray:
        return np.asarray(self._meta["bodies"][name]["world_quat"], dtype=float)

    def get_joint_position(self, name: str) -> float:
        start, width = self._offsets[name]
        return float(self._qpos[start]) if width == 1 else float(self._qpos[start])

    def get_qpos(self) -> np.ndarray:
        return self._qpos

    def get_self_contacts(self) -> dict:
        return {}

    def in_contact(self, _other: object) -> bool:
        return False


class _StubScene:
    def __init__(self, world: _StubWorld) -> None:
        self.entities = {"world": world}


def _load_success(config_path: Path) -> dict:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    success = data.get("success")
    assert isinstance(success, dict), f"{config_path.name}: missing success block"
    return success


def _load_world(config_path: Path) -> tuple[_StubScene, np.ndarray, dict]:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_cfg = data.get("task") or {}
    world_cfg = task_cfg.get("world") or {}
    meta_rel = world_cfg.get("meta_path")
    qpos_rel = world_cfg.get("qpos_path")
    assert meta_rel and qpos_rel, f"{config_path.name}: missing world block"
    meta = json.loads((REPO_ROOT / meta_rel).read_text(encoding="utf-8"))
    qpos = np.load(REPO_ROOT / qpos_rel)
    for joint, value in (world_cfg.get("qpos_overrides") or {}).items():
        offsets = _joint_q_offsets(meta)
        start, _width = offsets[joint]
        qpos = qpos.copy()
        qpos[start] = float(value)
    world = _StubWorld(meta, qpos)
    for name, record in (world_cfg.get("extra_sites") or {}).items():
        body, local = record[0], record[1]
        world._rc_sites[name] = (body, tuple(float(v) for v in local))
    info = {
        "rc_appliance": {
            cfg["name"]: bool(cfg.get("initial_on", False))
            for cfg in (task_cfg.get("appliances") or [])
        },
        "rc_timer": {},
    }
    return _StubScene(world), qpos, info


@pytest.mark.parametrize("suite", SUITES)
def test_compiled_robocasa_nop_fails(suite: str) -> None:
    """Every compiled RoboCasa task: success spec must be False at the frozen start."""
    from cloud_robotics_sim.core.success import create_success_condition

    configs = _suite_configs(suite)
    assert configs, f"{suite}: no compiled configs"
    failures = []
    for path in configs:
        success = _load_success(path)
        condition = create_success_condition(success)
        scene, _qpos, info = _load_world(path)
        if condition.evaluate(scene, robot=None, info=info) is not False:
            failures.append(path.name)
    assert not failures, (
        f"{suite}: {len(failures)} task(s) already succeed at the frozen start: "
        f"{failures}"
    )
