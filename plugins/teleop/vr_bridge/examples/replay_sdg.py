"""Replay recorded teleop episodes under domain randomization to synthesize data.

The "record once, synthesize many" half of the vr_bridge data loop:
``run_teleop.py --record`` captures one human demonstration as a
dreamdojo-layout HDF5 (joint-space actions); this script rebuilds the scene
in Genesis, replays the trajectory open-loop once per variant with object-pose
and camera-pose randomization, and re-renders multi-camera RGB/depth/seg.

This is the Genesis-side stand-in for Isaac Capture's episode-replay SDG
(replay recorded teleop episodes and render synthetic datasets), without
needing Isaac Sim.

Outputs:
  - ``--out``: dreamdojo-layout HDF5 (``episode_k`` groups: ``observations``
    uint8 video from the primary camera + ``actions`` float32 qpos), directly
    loadable by ``plugins/datasets/dreamdojo`` GenesisDataset.
  - ``--out-dir``: RoboTwin-layout per-variant HDF5 via
    ``cloud_robotics_sim.robotwin.recorder.EpisodeRecorder`` — every camera's
    rgb/depth/segmentation plus camera intrinsics/extrinsics.

Randomization notes: the Genesis rasterizer has no runtime light control
(``scene.add_light`` is BatchRenderer-only and the default rasterizer scene
lighting is fixed at build), so lighting jitter is out of scope here; per
variant we randomize prop poses (xy + yaw) and camera poses (pos + lookat).

Usage:
    python replay_sdg.py --input teleop.h5 --variants 8 --out sdg.h5
    python replay_sdg.py --input teleop.h5 --variants 2 --stride 5 \
        --max-frames 40 --out sdg.h5   # quick smoke run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Genesis-free helpers (importable by unit tests without genesis-world)
# ---------------------------------------------------------------------------


def load_source_actions(path: str | Path, episode: int = 0) -> np.ndarray:
    """Load the joint-space action track of one recorded episode.

    Accepts the dreamdojo layout written by ``TeleopRecorder.save_hdf5``
    (``episode_N/actions``, (T, D) float32).
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        key = f"episode_{episode}"
        if key not in h5:
            available = sorted(k for k in h5.keys() if k.startswith("episode_"))
            raise KeyError(f"{path} has no '{key}' (available: {available or 'none'})")
        actions = np.asarray(h5[key]["actions"], dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"episode actions must be (T, D), got {actions.shape}")
    return actions


def append_dreamdojo_episode(
    path: str | Path,
    observations: np.ndarray,
    actions: np.ndarray,
    task_name: str = "replay_sdg",
) -> Path:
    """Append one replayed episode to a dreamdojo-layout HDF5 file.

    Mirrors ``TeleopRecorder.save_hdf5`` so teleop-recorded and
    replay-synthesized episodes accumulate in the same format
    (``episode_N/observations`` (T, H, W, 3) uint8 + ``episode_N/actions``
    (T, D) float32).
    """
    observations = np.asarray(observations, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(f"observations must be (T, H, W, 3), got {observations.shape}")
    if actions.ndim != 2 or actions.shape[0] != observations.shape[0]:
        raise ValueError(
            f"actions must be (T, D) with T={observations.shape[0]}, "
            f"got {actions.shape}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as h5:
        n = sum(1 for key in h5.keys() if key.startswith("episode_"))
        group = h5.create_group(f"episode_{n}")
        group.create_dataset("observations", data=observations)
        group.create_dataset("actions", data=actions)
        group.attrs["task_name"] = task_name
    return path


def jitter_xy_yaw(
    rng: np.random.Generator,
    base_xy: tuple[float, float],
    xy_jitter: float,
    base_yaw: float = 0.0,
    yaw_jitter: float = np.pi,
) -> tuple[np.ndarray, float]:
    """Uniform xy jitter plus yaw jitter for a prop pose."""
    xy = np.asarray(base_xy, dtype=np.float64)
    if xy_jitter > 0.0:
        xy = xy + rng.uniform(-xy_jitter, xy_jitter, size=2)
    yaw = base_yaw + float(rng.uniform(-yaw_jitter, yaw_jitter))
    return xy, yaw


def jitter_camera(
    rng: np.random.Generator,
    pos: tuple[float, float, float],
    lookat: tuple[float, float, float],
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform jitter of a camera's position and lookat target."""
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    if jitter > 0.0:
        pos = pos + rng.uniform(-jitter, jitter, size=3)
        lookat = lookat + rng.uniform(-jitter, jitter, size=3)
    return pos, lookat


def yaw_to_quat(yaw: float) -> np.ndarray:
    """World-Z yaw angle (rad) to a wxyz quaternion."""
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def subsample_actions(actions: np.ndarray, stride: int, max_frames: int) -> np.ndarray:
    """Thin a (T, D) action track for quick smoke replays."""
    if stride > 1:
        actions = actions[::stride]
    if max_frames > 0:
        actions = actions[:max_frames]
    return actions


# ---------------------------------------------------------------------------
# Genesis scene
# ---------------------------------------------------------------------------

# Camera presets: name -> (pos, lookat). The first entry is the primary
# camera used for the dreamdojo ``observations`` stream.
DEFAULT_CAMERAS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "front": ((1.2, -1.0, 1.1), (0.2, 0.0, 0.45)),
    "side": ((0.1, 1.4, 1.0), (0.2, 0.0, 0.6)),
}

# Static scene dressing: base pose (x, y), half-extent z (size/2), size, color.
DEFAULT_PROPS: tuple[dict, ...] = (
    {"xy": (0.35, -0.15), "size": 0.06, "color": (0.8, 0.3, 0.2)},
    {"xy": (0.45, 0.10), "size": 0.05, "color": (0.2, 0.5, 0.8)},
    {"xy": (0.30, 0.22), "size": 0.045, "color": (0.9, 0.8, 0.3)},
)


def build_scene(
    resolution: int,
    device: str,
    cameras: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
):
    """Best-effort headless Genesis scene: ground + Franka + props + cameras."""
    import genesis as gs  # noqa: PLC0415

    try:  # project compat helper: CUDA -> CPU fallback, CI detection
        from cloud_robotics_sim.utils.genesis_compat import (  # noqa: PLC0415
            genesis_init,
        )

        genesis_init(headless=True, device=device)
    except ImportError:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    props = [
        scene.add_entity(
            gs.morphs.Box(
                size=(spec["size"],) * 3,
                pos=(spec["xy"][0], spec["xy"][1], spec["size"] / 2.0),
            ),
            surface=gs.surfaces.Default(color=spec["color"]),
            material=gs.materials.Rigid(rho=300.0, friction=0.5),
        )
        for spec in DEFAULT_PROPS
    ]
    cams = {
        name: scene.add_camera(
            res=(resolution, resolution),
            pos=pose[0],
            lookat=pose[1],
            fov=45,
            GUI=False,
        )
        for name, pose in cameras.items()
    }
    scene.build()
    return scene, robot, props, cams


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------


def run_replay(args: argparse.Namespace) -> None:
    """Build the scene and synthesize ``args.variants`` replayed episodes."""
    from cloud_robotics_sim.backends.genesis_backend import (  # noqa: PLC0415
        GenesisCameraBackend,
    )
    from cloud_robotics_sim.robotwin.recorder import EpisodeRecorder  # noqa: PLC0415

    actions = subsample_actions(
        load_source_actions(args.input, args.episode), args.stride, args.max_frames
    )
    n_frames, n_dofs = actions.shape
    if n_frames < 2:
        raise ValueError("need at least 2 action frames to replay")
    print(
        f"[replay_sdg] source: {n_frames} frames x {n_dofs} dofs "
        f"(episode {args.episode} of {args.input})"
    )

    cameras = dict(DEFAULT_CAMERAS)
    primary_cam = next(iter(cameras))
    scene, robot, props, cams = build_scene(args.resolution, args.device, cameras)
    hand = robot.get_link("hand")
    cam_backends = {name: GenesisCameraBackend(name, cam) for name, cam in cams.items()}
    rng = np.random.default_rng(args.seed)

    rich_dir = Path(args.out_dir) if args.out_dir else Path(args.out).with_suffix("")
    rich_dir = rich_dir.parent / f"{rich_dir.name}_rich"
    rich_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # fresh dataset file per run

    for variant in range(args.variants):
        recorder = EpisodeRecorder(
            task_name="replay_sdg",
            fps=args.fps,
            n_envs=1,
            metadata={
                "source": str(args.input),
                "source_episode": args.episode,
                "variant": variant,
                "seed": args.seed,
                "stride": args.stride,
            },
        )

        # --- per-variant randomization ---
        for name, backend in cam_backends.items():
            pos, lookat = jitter_camera(
                rng, cameras[name][0], cameras[name][1], args.camera_jitter
            )
            cams[name].set_pose(pos=pos, lookat=lookat)
            intrinsic, extrinsic = backend.get_camera_params()
            recorder.set_camera_params(name, intrinsic, extrinsic)
        for prop, spec in zip(props, DEFAULT_PROPS):
            xy, yaw = jitter_xy_yaw(rng, spec["xy"], args.object_jitter)
            prop.set_pos(np.array([xy[0], xy[1], spec["size"] / 2.0]))
            prop.set_quat(yaw_to_quat(yaw))

        # --- open-loop replay ---
        robot.set_qpos(actions[0], zero_velocity=True)
        for _ in range(args.warmup):
            scene.step()
        rgb_track = np.empty(
            (n_frames, args.resolution, args.resolution, 3), dtype=np.uint8
        )
        qpos_track = np.empty((n_frames, n_dofs), dtype=np.float32)
        for t in range(n_frames):
            robot.control_dofs_position(actions[t])
            scene.step()
            rgb_out, depth_out, seg_out = {}, {}, {}
            for name, backend in cam_backends.items():
                # Genesis 1.4 render() always returns the 4-tuple
                # (rgb, depth, segmentation, normal).
                rgb, depth, seg, _normal = backend.render(
                    rgb=True, depth=True, segmentation=True
                )
                rgb_out[name], depth_out[name], seg_out[name] = rgb, depth, seg
            qpos = np.asarray(robot.get_qpos(), dtype=np.float64).ravel()
            ee_pos = np.asarray(hand.get_pos(), dtype=np.float64).ravel()
            ee_quat = np.asarray(hand.get_quat(), dtype=np.float64).ravel()
            recorder.capture(
                t,
                rgb=rgb_out,
                depth=depth_out,
                segmentation=seg_out,
                qpos=qpos,
                endpose=np.concatenate([ee_pos, ee_quat]),
            )
            if primary_cam in rgb_out:
                rgb_track[t] = np.asarray(rgb_out[primary_cam], dtype=np.uint8)
            qpos_track[t] = qpos.astype(np.float32)

        rich_path = recorder.save_hdf5(rich_dir / f"variant_{variant:03d}.hdf5")
        if args.mp4:
            try:
                recorder.save_mp4(
                    rich_dir / f"variant_{variant:03d}.mp4", camera=primary_cam
                )
            except (OSError, RuntimeError) as exc:
                # Preview export spawns an ffmpeg subprocess; headless or
                # handle-starved environments (CI, some Windows shells) can
                # reject the spawn. The HDF5 data is already saved.
                print(f"[replay_sdg] WARNING: mp4 preview skipped: {exc}")
        append_dreamdojo_episode(
            out_path, rgb_track, qpos_track, task_name="replay_sdg"
        )
        print(
            f"[replay_sdg] variant {variant + 1}/{args.variants}: {n_frames} frames "
            f"-> {rich_path.name} (+ dreamdojo episode_{variant})"
        )

    print(f"[replay_sdg] dreamdojo dataset: {out_path}")
    print(f"[replay_sdg] rich per-variant files: {rich_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="dreamdojo-layout HDF5 recorded by run_teleop.py --record",
    )
    parser.add_argument("--episode", type=int, default=0, help="source episode index")
    parser.add_argument(
        "--variants", type=int, default=8, help="replayed episodes to synthesize"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("replay_sdg.h5"),
        help="dreamdojo-layout output dataset (rewritten each run)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for rich per-variant HDF5 (default: <out>_rich/)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--stride", type=int, default=1, help="replay every k-th source frame"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="cap replayed frames per variant (0 = all, after stride)",
    )
    parser.add_argument(
        "--object-jitter",
        type=float,
        default=0.05,
        help="prop xy jitter amplitude in meters (0 disables)",
    )
    parser.add_argument(
        "--camera-jitter",
        type=float,
        default=0.03,
        help="camera pos/lookat jitter amplitude in meters (0 disables)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="settle steps per variant"
    )
    parser.add_argument(
        "--device", default="cuda", help="genesis device ('cuda' falls back to cpu)"
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="also export a preview MP4 of the primary camera per variant",
    )
    args = parser.parse_args()
    run_replay(args)


if __name__ == "__main__":
    main()
