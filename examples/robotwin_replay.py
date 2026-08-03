"""Replay a RoboTwin demonstration inside Genesis.

This example loads a RoboTwin bridge (or synthesizes a short one for smoke
testing), composes a Genesis environment with the ALOHA-AgileX dual-arm robot,
and plays back the recorded trajectory frame by frame. Optionally renders a
video from a fixed camera.

Usage:
    uv run python examples/robotwin_replay.py --bridge data/trajectories/episode_0.bridge
    uv run python examples/robotwin_replay.py --headless --output-video outputs/robotwin_replay.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from cloud_robotics_sim import ComposerConfig, EnvironmentComposer
from cloud_robotics_sim.robotwin.bridge import (
    ObjectAsset,
    RobotwinBridge,
    RobotwinFrame,
)
from cloud_robotics_sim.robotwin.dual_arm_embodiment import (
    AlohaAgileX,
    AlohaAgileXConfig,
)
from cloud_robotics_sim.robotwin.replay_scene import RobotwinReplayScene
from cloud_robotics_sim.robotwin.replay_task import RobotwinReplayTask

logger = logging.getLogger(__name__)


def _make_synthetic_bridge(num_frames: int = 60) -> RobotwinBridge:
    """Create a tiny synthetic bridge for smoke testing without real data."""
    frames: list[RobotwinFrame] = []
    for i in range(num_frames):
        t = i / 20.0
        # Simple sinusoidal motion on a few joints.
        cmd = np.zeros(14, dtype=np.float64)
        cmd[0] = 0.3 * np.sin(2.0 * np.pi * t)
        cmd[6] = 0.04 + 0.02 * np.sin(2.0 * np.pi * t)
        cmd[7] = -0.3 * np.sin(2.0 * np.pi * t)
        cmd[13] = 0.04 + 0.02 * np.sin(2.0 * np.pi * t)

        frames.append(
            RobotwinFrame(
                timestamp=t,
                robot_command=cmd,
                robot_achieved_qpos=cmd.copy(),
                robot_base_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                robot_base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                object_states={
                    "can": {
                        "pos": np.array([0.5, 0.0, 0.74], dtype=np.float64),
                        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    }
                },
            )
        )

    return RobotwinBridge(
        task_name="synthetic_smoke_test",
        seed=0,
        fps=20.0,
        robot_urdf="assets/embodiments/aloha-agilex/urdf/robot.urdf",
        table_height=0.74,
        object_assets={
            "can": ObjectAsset(name="can", asset_type="mesh", path="assets/objects/can.glb")
        },
        frames=frames,
    )


def _load_or_make_bridge(path: str | None) -> RobotwinBridge:
    """Load a bridge from disk, or fall back to a synthetic one."""
    if path:
        bridge_path = Path(path)
        if bridge_path.exists():
            logger.info(f"Loading bridge from {bridge_path}")
            return RobotwinBridge.load(bridge_path)
        logger.warning(f"Bridge not found at {bridge_path}; using synthetic bridge")
    else:
        logger.info("No bridge path provided; using synthetic bridge")
    return _make_synthetic_bridge()


def main() -> int:
    """Run the RoboTwin replay example."""
    parser = argparse.ArgumentParser(
        description="Replay a RoboTwin demonstration in Genesis"
    )
    parser.add_argument(
        "--bridge",
        type=str,
        default=None,
        help="Path to a .bridge pickle file produced by RoboTwin bridge export.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive viewer.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="If set, render frames to this video file (requires headless or background).",
    )
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=[0.8, -0.8, 1.2],
        help="World-space camera position for rendering.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.74],
        help="World-space camera look-at point for rendering.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation timestep in seconds.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Physics substeps per simulation step.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Taichi device (cuda or cpu).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    bridge = _load_or_make_bridge(args.bridge)
    logger.info(
        f"Bridge: task={bridge.task_name}, frames={len(bridge)}, "
        f"fps={bridge.fps}, objects={list(bridge.object_assets.keys())}"
    )

    # Warn when the URDF is missing so the user gets a clear message.
    if not Path(bridge.robot_urdf).exists():
        logger.warning(
            f"Robot URDF not found: {bridge.robot_urdf}. "
            "The example will likely fail during spawn."
        )

    composer = EnvironmentComposer(
        ComposerConfig(
            dt=args.dt,
            substeps=args.substeps,
            headless=args.headless,
            resolution=(640, 480),
            backend="genesis",
            device=args.device,
        )
    )

    scene = RobotwinReplayScene(bridge)
    robot = AlohaAgileX(
        AlohaAgileXConfig(
            name="aloha_agilex",
            urdf_path=bridge.robot_urdf,
        )
    )
    task = RobotwinReplayTask(bridge)

    logger.info("Composing environment...")
    env = composer.compose(scene, robot, task)

    # Add a rendering camera if video output is requested.
    camera = None
    if args.output_video:
        try:
            camera = env.scene_backend.add_camera(
                name="replay_camera",
                pos=tuple(args.camera_pos),
                lookat=tuple(args.camera_lookat),
                resolution=(640, 480),
                fov=60.0,
            )
            logger.info(f"Added rendering camera: {camera.name}")
        except Exception as exc:
            logger.warning(f"Could not add rendering camera: {exc}")

    logger.info("Resetting environment and starting replay...")
    obs, info = env.reset(seed=bridge.seed)
    logger.info(f"Reset info: {info}")

    frames: list[np.ndarray] = []
    step_count = 0
    while True:
        if camera is not None:
            try:
                rendered = camera.render(rgb=True)
                if isinstance(rendered, tuple):
                    rendered = rendered[0]
                frames.append(np.asarray(rendered))
            except Exception as exc:
                logger.warning(f"Render failed at step {step_count}: {exc}")

        # The action is ignored by RobotwinReplayTask; pass zeros.
        action = np.zeros(robot.action_dim, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if step_count % 20 == 0:
            logger.info(f"Step {step_count} / frame {info.get('frame_index', '?')}")

        if terminated or truncated:
            break

    logger.info(f"Replay finished after {step_count} steps")
    env.close()

    if args.output_video and frames:
        try:
            import imageio

            output_path = Path(args.output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(output_path, frames, fps=bridge.fps)
            logger.info(f"Saved video to {output_path}")
        except Exception as exc:
            logger.error(f"Failed to save video: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
