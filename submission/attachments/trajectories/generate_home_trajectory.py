"""生成家庭隐形家务场景的机器人动作规划轨迹（预赛提交附件）。

使用 hierarchical_cuRobo_planner（D:\\githbi\\hierarchical_cuRobo_planner）的
WorkspaceAStar 高层规划器，在模拟家居台面场景中生成末端执行器避障路径。
A* 在 torch 后端运行（CPU/GPU 均可）；完整 IK/TO 关节空间轨迹优化需 CUDA/MUSA，
GPU 全管线结果见同目录 curobo_full_pipeline_result.npz / curobo_full_pipeline_demo.png。

运行方式：
    .venv/Scripts/python.exe submission/attachments/trajectories/generate_home_trajectory.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# 让本脚本无需安装即可导入规划器源码
PLANNER_ROOT = r"D:\githbi\hierarchical_cuRobo_planner"
sys.path.insert(0, PLANNER_ROOT)

from curobo_hierarchical_planner.planning.astar import WorkspaceAStar  # noqa: E402
from curobo_hierarchical_planner.types import WorkspaceConfig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 家庭台面整理场景：0.9m 高台面 + 一道横向收纳格挡板，机械臂从左侧取物翻越挡板到右侧归位
SCENE = {
    "bounds": [[-0.8, 0.8], [-0.8, 0.8], [0.0, 1.1]],
    "obstacles": [
        {"type": "cuboid", "name": "table", "dims": [1.6, 1.6, 0.04], "pose": [0.0, 0.0, 0.38]},
        {"type": "cuboid", "name": "shelf_divider", "dims": [0.20, 1.60, 0.36], "pose": [0.0, 0.0, 0.58]},
    ],
}
START = np.array([-0.55, -0.30, 0.45])  # 取物点（台面上方）
GOAL = np.array([0.55, 0.30, 0.45])  # 归位点（台面上方）


def main() -> None:
    world = WorkspaceConfig(
        bounds=np.array(SCENE["bounds"], dtype=np.float64),
        voxel_size=0.03,
        inflation_radius=0.05,  # 末端执行器安全包络
        connectivity=26,
        obstacles=SCENE["obstacles"],
    )
    planner = WorkspaceAStar(world)
    ee_path = planner.plan(START, GOAL, device="cpu")
    print(f"A* 规划成功：{ee_path.shape[0]} 个路径点")

    seg = np.diff(ee_path, axis=0)
    length = float(np.linalg.norm(seg, axis=1).sum())
    np.savez(
        os.path.join(OUT_DIR, "home_tidy_astar_path.npz"),
        ee_path=ee_path,
        start=START,
        goal=GOAL,
        path_length_m=length,
        voxel_size=world.voxel_size,
        inflation_radius=world.inflation_radius,
        obstacles=np.array([str(o) for o in SCENE["obstacles"]]),
    )
    print(f"路径长度 {length:.3f} m，已保存 home_tidy_astar_path.npz")
    _visualize(ee_path, world)


def _cuboid_edges(center: np.ndarray, half: np.ndarray):
    c, h = np.asarray(center, float), np.asarray(half, float)
    corners = np.array([[c[0] + i * h[0], c[1] + j * h[1], c[2] + k * h[2]]
                        for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [corners[[a, b]] for a, b in pairs]


def _visualize(ee_path: np.ndarray, world: WorkspaceConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], "-o",
            markersize=2, lw=1.5, label="A* EE path")
    ax.scatter(*START, c="b", s=80, label="start (pick)")
    ax.scatter(*GOAL, c="g", s=80, label="goal (place)")
    for obs in world.obstacles:
        c = np.asarray(obs["pose"], float)[:3]
        h = np.asarray(obs["dims"], float) / 2
        for edge in _cuboid_edges(c, h):
            ax.plot(*edge.T, "r-", lw=1.2)
    ax.plot([], [], "r-", label="obstacles")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Home tidying scene - workspace A* path (hierarchical_cuRobo_planner)")
    ax.legend()
    ax.view_init(elev=25, azim=-60)
    out = os.path.join(OUT_DIR, "home_tidy_astar_path.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"可视化已保存 {os.path.basename(out)}")


if __name__ == "__main__":
    main()
