"""Run the VR teleoperation bridge against a simulated robot.

Modes:
    --mock     FakeRobot with trivial kinematics (no genesis needed).
               The fake EE position is q[0:3], so you can watch the
               mock client move it.
    (default)  Real Genesis scene with a Franka arm, real IK, and an
               offscreen camera. Combine with --record to capture a
               dreamdojo-compatible HDF5 episode (M2 data loop).

Usage:
    python run_teleop.py --mock
    python run_teleop.py --config ../configs/vr_bridge.yaml
    python run_teleop.py --record episode.h5 --duration 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bridge import VRBridge  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "vr_bridge.yaml"


class _Pose:
    def __init__(self, pos, quat) -> None:
        self.pos = np.asarray(pos, dtype=np.float64)
        self.quat = np.asarray(quat, dtype=np.float64)


class FakeRobot:
    """Trivial 9-DoF stand-in: EE position is q[0:3], orientation fixed.

    Implements the duck-typed interface VRBridge expects from
    ``ArticulationBackend``.
    """

    def __init__(self, n_dofs: int = 9) -> None:
        self._q = np.zeros(n_dofs)

    def get_qpos(self) -> np.ndarray:
        return self._q.copy()

    def control_dofs_position(self, targets, stiffness=None, damping=None) -> None:
        arr = np.asarray(targets, dtype=np.float64)
        self._q[: arr.shape[0]] = arr

    def get_link_pose(self, link_name: str) -> _Pose:
        return _Pose(self._q[:3], [1.0, 0.0, 0.0, 0.0])

    def inverse_kinematics(self, link_name, pos, quat, dofs_idx=None):
        q = np.zeros(7)
        q[:3] = np.asarray(pos, dtype=np.float64)
        return q


def build_genesis_robot(with_camera: bool = True):
    """Best-effort Genesis Franka setup (requires genesis-world)."""
    import genesis as gs  # noqa: PLC0415

    try:  # project compat helper: CUDA -> CPU fallback, CI detection
        from cloud_robotics_sim.utils.genesis_compat import (  # noqa: PLC0415
            genesis_init,
        )

        genesis_init(headless=True, device="cuda")
    except ImportError:
        gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    camera = None
    if with_camera:
        camera = scene.add_camera(
            res=(224, 224),
            pos=(1.5, 0.0, 1.2),
            lookat=(0.1, 0.0, 0.85),
            fov=45,
            GUI=False,
        )
    scene.build()
    return scene, robot, camera


class _GenesisRobotAdapter:
    """Adapt a Genesis RigidEntity to the bridge's duck-typed interface."""

    def __init__(self, entity) -> None:
        self._entity = entity
        self._hand = entity.get_link("hand")

    def get_qpos(self) -> np.ndarray:
        return np.asarray(self._entity.get_qpos()).ravel()

    def control_dofs_position(self, targets, stiffness=None, damping=None) -> None:
        self._entity.control_dofs_position(np.asarray(targets))

    def get_link_pose(self, link_name: str) -> _Pose:
        link = self._entity.get_link(link_name)
        return _Pose(
            np.asarray(link.get_pos()).ravel(), np.asarray(link.get_quat()).ravel()
        )

    # Reject solutions whose position error exceeds 1 cm: Genesis returns a
    # best-effort pose even when the solver does not converge, and commanding
    # garbage joint targets is how teleop arms jump.
    IK_POS_ERR_TOL = 0.01

    def inverse_kinematics(self, link_name, pos, quat, dofs_idx=None):
        q, err = self._entity.inverse_kinematics(
            self._hand,
            pos=pos,
            quat=quat,
            init_qpos=self._entity.get_qpos(),  # warm start for per-tick tracking
            return_error=True,
        )
        err = np.asarray(err, dtype=np.float64).ravel()
        pos_err = float(np.linalg.norm(err[:3]))
        if pos_err > self.IK_POS_ERR_TOL:
            raise RuntimeError(f"IK did not converge: pos_err={pos_err:.4f} m")
        return np.asarray(q).ravel()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mock", action="store_true", help="use FakeRobot")
    parser.add_argument("--rate", type=float, default=60.0, help="sim tick rate")
    parser.add_argument("--duration", type=float, default=0.0, help="0 = forever")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="record an episode and save it as dreamdojo-compatible HDF5 "
        "(Genesis mode only; renders the camera every tick, so the loop "
        "runs at render speed)",
    )
    args = parser.parse_args()

    recording = args.record is not None and not args.mock
    if args.mock:
        robot = FakeRobot()
        scene, camera = None, None
        print("[bridge] using FakeRobot (mock mode)")
    else:
        scene, entity, camera = build_genesis_robot(with_camera=recording)
        robot = _GenesisRobotAdapter(entity)

    bridge = VRBridge.from_yaml(args.config, robot=robot)
    bridge.start()
    print(
        f"[bridge] listening udp={bridge.transport.state_port} "
        f"tcp={bridge.transport.control_port} — start mock_client.py now"
    )

    if recording:
        bridge.recorder.start()
        print(f"[bridge] recording from tick 0 -> {args.record}")

    dt = 1.0 / args.rate
    t0 = time.monotonic()
    last_print = 0.0
    try:
        while True:
            t = time.monotonic() - t0
            if args.duration and t > args.duration:
                break
            rgb = None
            if recording and camera is not None:
                rgb, *_ = camera.render(rgb=True)
            info = bridge.step(dt, rgb=rgb)
            if scene is not None:
                scene.step()
            if t - last_print > 1.0:
                last_print = t
                q = robot.get_qpos()
                print(
                    f"[{t:6.1f}s] safety={info['safety_state']:<12} "
                    f"age={info['age_ms']:6.1f}ms rec={info['recording']} "
                    f"ee=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}) "
                    f"loss={bridge.mailbox.loss_rate:.1%}"
                )
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if recording and bridge.recorder.n_frames:
            bridge.recorder.stop()
            saved = bridge.recorder.save_hdf5(args.record)
            print(
                f"[bridge] episode saved: {saved} ({bridge.recorder.n_frames} frames)"
            )
        bridge.stop()
        print("[bridge] stopped")


if __name__ == "__main__":
    main()
