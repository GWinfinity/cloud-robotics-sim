"""Replay HiFi-UMI-2K episodes in Genesis with a Franka Panda and report tracking deviation.

Pipeline: LeRobot v3 parquet -> 125 Hz resample -> recenter into the robot
workspace -> per-frame IK tracking -> achieved-vs-target deviation report
(JSON + Markdown + PNG).

Usage (from repo root):

    uv run python -m examples.hifiumi.replay_hifiumi \
        --episode 0 --hand right --rate 125

Requires: genesis-world (installed), pyarrow, matplotlib, and a downloaded
HiFi-UMI-2K part directory (see ``data/hifiumi/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cloud_robotics_sim.backends.genesis_backend import GenesisBackend  # noqa: E402
from examples.hifiumi.hifiumi_loader import (  # noqa: E402
    UMIEpisode,
    load_episode,
    quat_angle_deg,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    recenter_to_workspace,
    resample_trajectory,
)

logger = logging.getLogger("hifiumi_replay")

EE_LINK = "hand"
FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
ARM_DOFS = 7
FRANKA_MAX_WIDTH_M = 0.08  # both fingers combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _summarize(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _trajectory_stats(episode: UMIEpisode, hand_name: str) -> dict:
    hand = getattr(episode, hand_name)
    dt = np.diff(episode.timestamps)
    extent = np.ptp(hand.pos, axis=0)
    seg = np.linalg.norm(np.diff(hand.pos, axis=0), axis=1)
    return {
        "source_fps": episode.fps,
        "n_frames_raw": len(episode.timestamps),
        "duration_s": episode.duration_s,
        "dt_mean_ms": float(np.mean(dt) * 1e3) if len(dt) else 0.0,
        "dt_jitter_std_ms": float(np.std(dt) * 1e3) if len(dt) else 0.0,
        "hand_valid_ratio": float(np.mean(hand.valid)),
        "frame_valid_ratio": float(np.mean(episode.frame_valid)),
        "workspace_extent_m": [float(v) for v in extent],
        "path_length_m": float(np.sum(seg)),
        "gripper_range_rad": [float(np.min(hand.gripper)), float(np.max(hand.gripper))],
    }


def _write_report(
    out_dir: Path,
    report: dict,
    times: np.ndarray,
    pos_err_mm: np.ndarray,
    rot_err_deg: np.ndarray,
    target_pos: np.ndarray,
    achieved_pos: np.ndarray,
) -> None:
    """Write report.json / report.md and the deviation plot to ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(times, pos_err_mm, lw=0.8)
    ax.axhline(
        report["thresholds"]["pos_tolerance_mm"],
        color="r",
        ls="--",
        lw=0.8,
        label="3 mm tolerance",
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("EE position error [mm]")
    ax.legend()
    ax.set_title("Position tracking error")

    ax = axes[0, 1]
    ax.plot(times, rot_err_deg, lw=0.8, color="tab:orange")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("EE rotation error [deg]")
    ax.set_title("Rotation tracking error")

    ax = axes[1, 0]
    ax.plot(target_pos[:, 0], target_pos[:, 1], lw=1.0, label="target")
    ax.plot(achieved_pos[:, 0], achieved_pos[:, 1], lw=0.8, ls="--", label="achieved")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view (xy)")
    ax.legend()
    ax.axis("equal")

    ax = axes[1, 1]
    ax.plot(target_pos[:, 2], lw=1.0, label="target z")
    ax.plot(achieved_pos[:, 2], lw=0.8, ls="--", label="achieved z")
    ax.set_xlabel("frame")
    ax.set_ylabel("z [m]")
    ax.legend()
    ax.set_title("Height profile")

    fig.tight_layout()
    fig.savefig(out_dir / "deviation_report.png", dpi=120)
    plt.close(fig)

    md = [
        f"# HiFi-UMI 回放偏差报告 — episode {report['episode']} ({report['hand']} hand)",
        "",
        f"- 任务：`{report['task']}`",
        f"- 回放速率：{report['replay_rate_hz']:.0f} Hz（源 {report['source']['source_fps']:.0f} fps 重采样）",
        f"- 帧数：{report['n_frames']}（原始 {report['source']['n_frames_raw']}）",
        f"- 位置误差 mm：mean {report['pos_err_mm']['mean']:.2f} / p95 {report['pos_err_mm']['p95']:.2f} / max {report['pos_err_mm']['max']:.2f}",
        f"- 旋转误差 deg：mean {report['rot_err_deg']['mean']:.2f} / p95 {report['rot_err_deg']['p95']:.2f} / max {report['rot_err_deg']['max']:.2f}",
        f"- 可达帧占比：{report['reachable_frac']*100:.1f}%（可达阈值 {report['thresholds']['reach_threshold_mm']:.0f} mm）；"
        f"可达帧内 ≤3mm 占比：{(report['frac_within_pos_tolerance_reachable'] or 0)*100:.1f}%",
        f"- 全轨迹 ≤3mm 帧占比：{report['frac_within_pos_tolerance']*100:.1f}%（阈值 {report['thresholds']['pos_tolerance_mm']:.0f} mm）",
        f"- 吞吐：{report['throughput']['steps_per_s']:.0f} steps/s，实时倍率 {report['throughput']['real_time_factor']:.2f}x",
        "",
        "![deviation](deviation_report.png)",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_episode(args: argparse.Namespace) -> dict:
    """Replay one HiFi-UMI episode on a Franka arm and return the deviation report."""
    episode = load_episode(args.data_root, args.episode)
    hand_name = args.hand
    hand = getattr(episode, hand_name)
    logger.info(
        "Episode %d: task=%r, %d frames @ %.0f fps, valid(hand)=%.1f%%",
        episode.episode_index,
        episode.task,
        len(episode.timestamps),
        episode.fps,
        100 * np.mean(hand.valid),
    )

    # Keep only frames valid for this hand (carry-forward is dataset policy for
    # invalid frames; we drop them for an honest replay).
    keep = hand.valid & episode.frame_valid
    hand.pos, hand.quat, hand.gripper = (
        hand.pos[keep],
        hand.quat[keep],
        hand.gripper[keep],
    )
    timestamps = episode.timestamps[keep]

    t_out, pos, quat, gripper = resample_trajectory(hand, timestamps, args.rate)
    if args.anchor_mode == "centroid":
        # Anchor the trajectory centroid at the workspace sweet spot — better
        # coverage for large-motion human demos than anchoring the first frame.
        pos = pos - pos.mean(axis=0) + np.asarray(args.anchor, dtype=np.float64)
    else:
        pos, quat = recenter_to_workspace(pos, quat, np.array(args.anchor))
    if args.max_frames:
        t_out, pos, quat, gripper = (
            t_out[: args.max_frames],
            pos[: args.max_frames],
            quat[: args.max_frames],
            gripper[: args.max_frames],
        )
    n_frames = len(t_out)

    gripper_max = float(np.max(np.abs(gripper))) or 1.0

    # --- Genesis setup ---
    backend = GenesisBackend()
    backend.initialize(headless=True, device=args.device)
    scene = backend.create_scene(
        dt=1.0 / args.rate, substeps=args.substeps, headless=True
    )
    robot = backend.load_mjcf(FRANKA_MJCF, pos=(0.0, 0.0, 0.0))
    scene.add_articulation(robot)
    scene.build(n_envs=1)

    # IK bootstrap at the first target; derive the constant EE-frame offset so
    # that orientation tracking is relative (UMI fingertip frame != Franka hand frame).
    q0 = robot.inverse_kinematics(EE_LINK, pos[0], quat=None)
    robot.set_qpos(q0)
    scene.step()
    q_ee0 = robot.get_link_pose(EE_LINK).quat
    offset = quat_mul_wxyz(q_ee0, quat_conjugate_wxyz(quat[0]))
    target_quat = quat_mul_wxyz(np.broadcast_to(offset, quat.shape), quat)

    achieved_pos = np.zeros((n_frames, 3))
    achieved_quat = np.zeros((n_frames, 4))
    pos_err_mm = np.zeros(n_frames)
    rot_err_deg = np.zeros(n_frames)
    ik_fail = 0

    t_wall0 = time.perf_counter()
    for i in range(n_frames):
        # Kinematic retarget replay: IK-solve each target pose and set qpos
        # directly. Deviation then measures pure retarget fidelity (the metric
        # comparable to HiFi-UMI's 3 mm claim), not PD-controller lag.
        try:
            q = robot.inverse_kinematics(EE_LINK, pos[i], target_quat[i])
        except Exception:
            ik_fail += 1
            q = None
        if q is not None:
            cmd = np.asarray(q, dtype=np.float64).copy()
            width = min(abs(gripper[i]) / gripper_max, 1.0) * FRANKA_MAX_WIDTH_M
            if cmd.shape[0] >= ARM_DOFS + 2:
                cmd[ARM_DOFS : ARM_DOFS + 2] = width / 2.0
            robot.set_qpos(cmd)
        scene.step()
        pose = robot.get_link_pose(EE_LINK)
        achieved_pos[i] = pose.pos
        achieved_quat[i] = pose.quat
        pos_err_mm[i] = np.linalg.norm(pose.pos - pos[i]) * 1e3
        dq = quat_mul_wxyz(quat_conjugate_wxyz(target_quat[i]), pose.quat)
        rot_err_deg[i] = quat_angle_deg(dq)
    wall_s = time.perf_counter() - t_wall0

    sim_s = n_frames / args.rate
    # Reachability split: frames far beyond the arm's workspace dominate the
    # mean; report retarget fidelity on reachable frames separately.
    reachable = pos_err_mm <= args.reach_threshold_mm
    report = {
        "episode": episode.episode_index,
        "task": episode.task,
        "hand": hand_name,
        "replay_rate_hz": args.rate,
        "n_frames": n_frames,
        "anchor_pos_m": list(map(float, args.anchor)),
        "anchor_mode": args.anchor_mode,
        "source": _trajectory_stats(episode, hand_name),
        "pos_err_mm": _summarize(pos_err_mm),
        "rot_err_deg": _summarize(rot_err_deg),
        "reachable_frac": float(np.mean(reachable)),
        "pos_err_mm_reachable": (
            _summarize(pos_err_mm[reachable]) if reachable.any() else None
        ),
        "rot_err_deg_reachable": (
            _summarize(rot_err_deg[reachable]) if reachable.any() else None
        ),
        "thresholds": {
            "pos_tolerance_mm": args.tolerance_mm,
            "reach_threshold_mm": args.reach_threshold_mm,
        },
        "frac_within_pos_tolerance": float(np.mean(pos_err_mm <= args.tolerance_mm)),
        "frac_within_pos_tolerance_reachable": (
            float(np.mean(pos_err_mm[reachable] <= args.tolerance_mm))
            if reachable.any()
            else None
        ),
        "ik_failures": ik_fail,
        "throughput": {
            "wall_s": wall_s,
            "sim_s": sim_s,
            "steps_per_s": n_frames / wall_s if wall_s > 0 else 0.0,
            "real_time_factor": sim_s / wall_s if wall_s > 0 else 0.0,
        },
    }
    out_dir = Path(args.out) / f"episode{episode.episode_index:04d}_{hand_name}"
    _write_report(
        out_dir, report, t_out - t_out[0], pos_err_mm, rot_err_deg, pos, achieved_pos
    )
    logger.info("Report written to %s", out_dir)
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        default="data/hifiumi/chunk-0000/part-0000",
        help="HiFi-UMI-2K part directory",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--rate", type=float, default=125.0, help="Replay rate in Hz")
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        default=[0.40, 0.0, 0.45],
        help="EE anchor point in robot workspace [m]",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["start", "centroid"],
        default="centroid",
        help="Anchor first frame or trajectory centroid",
    )
    parser.add_argument(
        "--reach-threshold-mm",
        type=float,
        default=20.0,
        help="Frames with pos err above this are counted as unreachable",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=3.0,
        help="Position tolerance for the pass metric",
    )
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="Genesis device: cpu | cuda")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit replay frames (0 = full episode)",
    )
    parser.add_argument("--out", default="outputs/hifiumi_replay")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    report = replay_episode(args)
    print(
        json.dumps(
            {
                "pos_err_mm": report["pos_err_mm"],
                "frac_within_pos_tolerance": report["frac_within_pos_tolerance"],
                "throughput": report["throughput"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
