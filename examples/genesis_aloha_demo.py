"""Minimal RoboTwin->Genesis migration skeleton (migration doc section 10).

Demonstrates the P0 migration deliverables end to end on the Genesis backend:

- URDF loading with ``fixed=True`` (section 4.1 checklist),
- PD gain mapping via ``set_dofs_gains`` (section 2, config.yml mapping),
- single/multi-link IK and OMPL ``plan_path`` (replacing mplib RRT, 5.2),
- camera intrinsics/extrinsics capture (section 7),
- HDF5 + MP4 episode export via ``EpisodeRecorder`` (sections 7/10),
- friction domain randomization (section 2, DR row).

Usage:
    # Synthetic arm (no assets required, runs on CPU):
    uv run python examples/genesis_aloha_demo.py --out outputs/genesis_aloha_demo

    # Converted aloha-agilex URDF (after tools/convert_assets.py):
    uv run python examples/genesis_aloha_demo.py \
        --urdf assets_genesis/embodiments/aloha-agilex/robot.urdf \
        --left-ee left_gripper_base --right-ee right_gripper_base \
        --render --out outputs/genesis_aloha_demo
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from cloud_robotics_sim.backends.genesis_backend import GenesisBackend
from cloud_robotics_sim.robotwin.curobo_planner import (
    CuRoboPlannerConfig,
    CuRoboPlannerUnavailableError,
    HierarchicalCuRoboPlanner,
    PlannerError,
)
from cloud_robotics_sim.robotwin.recorder import EpisodeRecorder

logger = logging.getLogger(__name__)

SYNTHETIC_ARM_URDF = """<?xml version="1.0"?>
<robot name="synthetic_arm">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <mass value="0.5"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <mass value="0.3"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <link name="ee_link">
    <inertial>
      <mass value="0.1"/><origin xyz="0 0 0.05"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
  <joint name="ee_fixed" type="fixed">
    <parent link="link2"/><child link="ee_link"/>
    <origin xyz="0 0 0.15"/>
  </joint>
</robot>
"""


def _write_synthetic_arm(out_dir: Path) -> Path:
    """Write the fallback synthetic fixed-base arm URDF."""
    urdf_path = out_dir / "synthetic_arm" / "robot.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(SYNTHETIC_ARM_URDF, encoding="utf-8")
    return urdf_path


def main() -> int:
    """Run the migration-skeleton demo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf", type=str, default=None, help="Robot URDF path.")
    parser.add_argument("--left-ee", type=str, default=None, help="Left EE link name.")
    parser.add_argument(
        "--right-ee", type=str, default=None, help="Right EE link name."
    )
    parser.add_argument(
        "--base-link", type=str, default="base_link", help="Robot base link name."
    )
    parser.add_argument(
        "--planner",
        choices=["auto", "ompl", "curobo"],
        default="auto",
        help="Motion planner: 'ompl' = Genesis plan_path; 'curobo' = hierarchical "
        "cuRobo planner (requires the external package + CUDA/MUSA); 'auto' = "
        "try cuRobo first, fall back to OMPL (doc section 5.2).",
    )
    parser.add_argument("--steps", type=int, default=30, help="Steps per waypoint.")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording fps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Attach a head camera and record RGB/depth (requires rendering support).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/genesis_aloha_demo",
        help="Output directory for episode files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 0. Resolve robot URDF (fixed base, section 4.1 checklist) ----------
    if args.urdf:
        urdf_path = Path(args.urdf)
        if not urdf_path.is_file():
            logger.error("URDF not found: %s", urdf_path)
            return 2
    else:
        urdf_path = _write_synthetic_arm(out_dir)
        logger.info("No --urdf given; using synthetic arm at %s", urdf_path)

    # ---------- 1. Backend init ----------
    backend = GenesisBackend()
    backend.initialize(headless=True, device="cpu")
    scene = backend.create_scene(dt=1.0 / 100.0, substeps=1, headless=True)

    robot = backend.load_urdf(str(urdf_path), pos=(0.0, 0.0, 0.0), fixed=True)
    scene.add_articulation(robot)

    camera = None
    if args.render and scene.renderer is not None:
        camera = scene.renderer.add_camera(
            name="head",
            pos=(0.0, -0.8, 0.6),
            lookat=(0.0, 0.0, 0.2),
            resolution=(640, 480),
            fov=60.0,
        )

    scene.build()
    logger.info(
        "Loaded %s (%d DoFs, fixed_base=%s)",
        urdf_path.name,
        robot.n_dofs,
        robot.is_fixed_base(),
    )

    # ---------- 2. PD gains (section 2: config.yml stiffness/damping) ----------
    kp = np.full(robot.n_dofs, 50.0)
    kv = np.full(robot.n_dofs, 5.0)
    robot.set_dofs_gains(kp, kv)

    # ---------- 3. Goal pose ----------
    down_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # gripper down
    dual_arm = bool(args.left_ee and args.right_ee)
    if dual_arm:
        ee_links = [args.left_ee, args.right_ee]
        goal_pos = np.array([0.3, 0.15, 0.25], dtype=np.float64)
        goal_quat: np.ndarray | None = down_quat
    else:
        ee_links = [args.left_ee or "link2"]
        goal_pos = np.array([0.0, 0.15, 0.3], dtype=np.float64)
        goal_quat = None

    # ---------- 4. Planning (doc 5.2: cuRobo first, OMPL fallback) ----------
    traj: np.ndarray | None = None
    if args.planner != "ompl":
        planner = HierarchicalCuRoboPlanner(
            CuRoboPlannerConfig(
                urdf_path=str(urdf_path),
                base_link=args.base_link,
                ee_link=ee_links[0],
            )
        )
        try:
            start = np.asarray(robot.get_qpos()).reshape(-1)[: robot.n_dofs]
            traj = planner.plan_to_ee_pose(start, goal_pos, goal_quat)
            logger.info("Planned with hierarchical_curobo (%d waypoints)", len(traj))
        except (CuRoboPlannerUnavailableError, PlannerError) as exc:
            if args.planner == "curobo":
                logger.error("cuRobo planner requested but failed: %s", exc)
                return 3
            logger.warning("cuRobo planner unavailable; using OMPL (%s)", exc)

    if traj is None:
        if dual_arm:
            q_goal = robot.inverse_kinematics_multilink(
                ee_links,
                poss=[goal_pos, goal_pos * np.array([1.0, -1.0, 1.0])],
                quats=[down_quat, down_quat],
            )
        else:
            q_goal = robot.inverse_kinematics(ee_links[0], pos=goal_pos, quat=goal_quat)
        try:
            traj = robot.plan_path(q_goal, num_waypoints=20)
        except Exception as exc:  # planning is best-effort for skeleton assets
            logger.warning("plan_path failed (%s); interpolating linearly", exc)
            traj = np.linspace(robot.get_qpos()[: robot.n_dofs], q_goal, num=20)

    # ---------- 5. Recorder + camera params (sections 7/10) ----------
    rec = EpisodeRecorder(task_name="genesis_aloha_demo", fps=args.fps)
    if camera is not None:
        rec.set_camera_params("head", *camera.get_camera_params())

    for step_i, waypoint in enumerate(traj):
        robot.control_dofs_position(np.asarray(waypoint, dtype=np.float64))
        for _ in range(max(1, args.steps // len(traj))):
            scene.step()
        ee_pose = robot.get_link_pose(ee_links[0])
        rgb, depth = None, None
        if camera is not None:
            try:
                rgb_arr, depth_arr = camera.render(rgb=True, depth=True)
                rgb, depth = {"head": rgb_arr}, {"head": depth_arr}
            except Exception as exc:
                logger.warning("render skipped at step %d: %s", step_i, exc)
        rec.capture(
            step_i,
            rgb=rgb,
            depth=depth,
            qpos=robot.get_qpos()[: robot.n_dofs],
            endpose=np.concatenate([ee_pose.pos, ee_pose.quat]),
        )

    # ---------- 6. Friction DR example (section 2) ----------
    try:
        rng = np.random.default_rng(args.seed)
        robot.set_friction_ratio(0.8 + 0.4 * rng.random(robot.n_dofs))
    except Exception as exc:
        logger.warning("friction DR skipped: %s", exc)

    # ---------- 7. Export HDF5 (+ MP4) ----------
    hdf5_path = rec.save_hdf5(out_dir / "episode_000.hdf5")
    logger.info("Saved episode: %s (%d frames)", hdf5_path, rec.n_frames)
    if camera is not None:
        try:
            mp4_path = rec.save_mp4(out_dir / "episode_000.mp4", camera="head")
            logger.info("Saved video: %s", mp4_path)
        except Exception as exc:
            logger.warning("MP4 export skipped: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
