"""XPBD 绳缆 + 稀疏 SDF 碰撞演示：绳子披覆到球障碍物上。

场景：地面 + 球障碍物；绳竖直悬挂于球左侧，钉随后被向右拖动
（smoothstep），绳子被拉过球顶形成披覆，全程与 SDF 场做碰撞投影。
碰撞场二选一：

* ``--backend xuvdb`` —— trimesh 网格经 ``meshes_to_sdf`` 烘成窄带
  level set（xuvdb kernel 侧三线性采样 + ``sdf_normal`` 法向）；
* ``--backend analytic`` —— 纯 torch 解析场（地面/球并集），无 xuvdb
  依赖，CPU 可跑；
* ``--backend auto`` —— 有 xuvdb 用 xuvdb，否则解析。

输出（默认 ``outputs/cable_sim/``）：
轨迹 ``trajectory.npz``、侧视动画 ``rope_drape.gif``、打印每步耗时与
最终诊断（最大穿透、拉伸率、动能）。

用法::

    uv run python examples/cable_sim/xpbd_rope_demo.py --backend auto
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cloud_robotics_sim.vdb import (
    XpbdRope,
    XuvdbSdfField,
    ground_field,
    has_xuvdb,
    sphere_field,
    union_fields,
)

# 场景几何（世界单位：米）
PIN = (-0.05, 0.0, 1.50)  # 钉初始在球左侧，随后向右拖动使绳披覆球顶
PIN_DRAG_TARGET_X = 0.70
PIN_DRAG_TIME = 2.5  # 拖动时长（s），smoothstep 加减速
N_PARTICLES = 50
SEGMENT = 0.028  # 绳长 1.372：垂落尖端 z≈0.13，离地、贴球左侧都不相交
SPHERE_CENTER = (0.45, 0.0, 0.25)
SPHERE_RADIUS = 0.25
MARGIN = 0.004
VOXEL = 0.004


def build_field(backend: str):
    """按后端构造碰撞场，返回 (field, backend_info)。"""
    if backend == "analytic" or (backend == "auto" and not has_xuvdb()):
        field = union_fields(
            ground_field(0.0),
            sphere_field(SPHERE_CENTER, SPHERE_RADIUS),
        )
        return field, "analytic (torch)"

    from cloud_robotics_sim.vdb.core.static_scene import meshes_to_sdf

    t0 = time.perf_counter()
    import trimesh

    ground = trimesh.creation.box(
        extents=(1.6, 0.6, 0.1),
        transform=_translate(0.1, 0.0, -0.05),
    )
    ball = trimesh.creation.icosphere(radius=SPHERE_RADIUS, subdivisions=3)
    ball.apply_transform(_translate(*SPHERE_CENTER))
    vol = meshes_to_sdf(
        [
            (np.asarray(ground.vertices), np.asarray(ground.faces)),
            (np.asarray(ball.vertices), np.asarray(ball.faces)),
        ],
        voxel_size=VOXEL,
        band=3.0,
        name="rope_demo_scene",
    )
    bake_s = time.perf_counter() - t0
    report = vol.memory_report()
    print(
        f"[bake] meshes_to_sdf: {bake_s:.2f}s, {report['n_leaves']} leaves, "
        f"{report['active_voxels']} active voxels"
    )
    return XuvdbSdfField(vol), f"xuvdb (voxel={VOXEL})"


def _translate(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["auto", "xuvdb", "analytic"], default="auto"
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--device", default=None, help="cuda / musa / cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/cable_sim"))
    args = parser.parse_args()

    field, field_info = build_field(args.backend)

    rope = XpbdRope(
        N_PARTICLES,
        SEGMENT,
        origin=PIN,
        direction=(0.0, 0.0, -1.0),
        mass=0.008,
        margin=MARGIN,
        friction=0.6,
        # 演示靠"拖动钉"制造披覆；弯曲约束保持默认关闭（软绳）。
        bend_compliance=-1.0,
        linear_damping=0.5,
        substeps=8,
        solver_iterations=4,
        device=args.device or "cpu",
    )
    rope.pin(0)

    drag_steps = int(PIN_DRAG_TIME / args.dt)

    def pin_x(step: int) -> float:
        s = min(step / max(drag_steps, 1), 1.0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep 加减速
        return PIN[0] + (PIN_DRAG_TARGET_X - PIN[0]) * s

    print(
        f"[demo] field={field_info} particles={rope.n_particles} "
        f"length={rope.rest_length:.3f} device={rope.device}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    traj = np.empty((args.steps, rope.n_particles, 3), dtype=np.float32)
    diag_hist: list[dict[str, float]] = []

    step_times = np.empty(args.steps)
    t0 = time.perf_counter()
    for i in range(args.steps):
        t_step = time.perf_counter()
        if i < drag_steps:
            rope.pin(0, (pin_x(i), 0.0, PIN[2]))
        rope.step(args.dt, field=field)
        step_times[i] = time.perf_counter() - t_step
        traj[i] = rope.positions.detach().cpu().numpy()
        diag_hist.append(rope.diagnostics(field))
    total_s = time.perf_counter() - t0

    final = diag_hist[-1]
    print(f"[demo] {args.steps} steps in {total_s:.2f}s ({args.steps/total_s:.0f} fps)")
    print(
        f"[demo] per-step: mean={step_times.mean()*1e3:.2f}ms "
        f"max={step_times.max()*1e3:.2f}ms "
        f"(substeps×iters={rope.substeps}×{rope.solver_iterations})"
    )
    print(
        f"[demo] final: max_penetration={final['max_penetration']*1e3:.3f}mm "
        f"stretch={final['stretch_ratio']*100:.2f}% "
        f"kinetic_energy={final['kinetic_energy']:.2e}J "
        f"tip_z={float(rope.positions[-1, 2]):.3f}"
    )

    np.savez(
        args.out / "trajectory.npz",
        traj=traj,
        dt=args.dt,
        pin=np.asarray(PIN),
        sphere_center=np.asarray(SPHERE_CENTER),
        sphere_radius=SPHERE_RADIUS,
        step_time_ms=step_times * 1e3,
    )
    gif_path = _render_gif(traj, args.out / "rope_drape.gif")
    print(f"[demo] wrote {args.out / 'trajectory.npz'} and {gif_path}")


def _render_gif(traj: np.ndarray, path: Path, n_frames: int = 60) -> Path:
    """侧视 (x-z) 动画：地面、球、绳。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    xlim = (PIN[0] - 0.25, SPHERE_CENTER[0] + SPHERE_RADIUS + 0.35)
    zlim = (-0.05, PIN[2] + 0.1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    # 地面与球（静态）
    ax.axhline(0.0, color="saddlebrown", lw=2)
    ball = plt.Circle(
        (SPHERE_CENTER[0], SPHERE_CENTER[2]),
        SPHERE_RADIUS,
        color="steelblue",
        alpha=0.4,
    )
    ax.add_patch(ball)
    (line,) = ax.plot([], [], "o-", color="tomato", lw=2, ms=4)
    (pin_m,) = ax.plot([], [], "ks", ms=6)

    idx = np.linspace(0, len(traj) - 1, n_frames).astype(int)

    def update(fi: int):
        i = idx[fi]
        line.set_data(traj[i, :, 0], traj[i, :, 2])
        pin_m.set_data([traj[i, 0, 0]], [traj[i, 0, 2]])
        return (line, pin_m)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
