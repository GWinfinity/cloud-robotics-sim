"""Shared fixtures for vr_bridge tests (no headset / no Genesis required)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make ``plugins/teleop`` importable so ``vr_bridge.core.*`` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Make ``plugins/datasets`` importable so cross-plugin tests can load
# teleop HDF5 output with ``dreamdojo.core.dataset.GenesisDataset``.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "datasets"))

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MAPPINGS_DIR = PLUGIN_ROOT / "configs" / "mappings"
MAIN_CONFIG = PLUGIN_ROOT / "configs" / "vr_bridge.yaml"


def make_hand(
    pos=(0.3, 0.0, 0.5),
    quat=(1.0, 0.0, 0.0, 0.0),
    trigger=0.0,
    grip=0.0,
    thumbstick=(0.0, 0.0),
) -> dict:
    """Build one hand_state dict matching protocol v1."""
    return {
        "pos": list(pos),
        "quat": list(quat),
        "trigger": trigger,
        "grip": grip,
        "thumbstick": list(thumbstick),
    }


def make_state_dict(seq=0, client_time_ms=0, left=None, right=None, head=None) -> dict:
    """Build one state datagram dict matching protocol v1."""
    state = {
        "type": "state",
        "seq": seq,
        "client_time_ms": client_time_ms,
        "left": left if left is not None else make_hand(),
        "right": right if right is not None else make_hand(),
    }
    if head is not None:
        state["head"] = head
    return state


class StubRobot:
    """Duck-typed articulation stand-in for ``VRBridge``.

    ``inverse_kinematics`` encodes the requested pose into the returned
    joint vector (q[:3] = pos, q[3:7] = quat) so tests can verify which
    target pose actually reached the IK stage.
    """

    def __init__(self, n_dofs: int = 9, ee_pos=(0.3, 0.0, 0.5)) -> None:
        self._q = np.zeros(n_dofs, dtype=np.float64)
        self._ee_pos = np.asarray(ee_pos, dtype=np.float64)
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.control_calls: list[np.ndarray] = []
        self.ik_calls: list[tuple[str, np.ndarray, np.ndarray]] = []

    def get_qpos(self) -> np.ndarray:
        return self._q.copy()

    def control_dofs_position(self, arr) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        self.control_calls.append(arr.copy())
        self._q[: arr.shape[0]] = arr

    def get_link_pose(self, name: str):
        return SimpleNamespace(pos=self._ee_pos.copy(), quat=self._ee_quat.copy())

    def inverse_kinematics(self, link_name: str, pos, quat) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        self.ik_calls.append((link_name, pos.copy(), quat.copy()))
        q = np.zeros(7, dtype=np.float64)
        q[:3] = pos
        q[3:7] = quat
        return q


@pytest.fixture()
def stub_robot() -> StubRobot:
    """A fresh 9-dof stub robot for each test."""
    return StubRobot()
