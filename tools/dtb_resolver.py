#!/usr/bin/env python3
"""Resolve the DexToolBench EFE suite to the compile knowledge base.

Merges the composed world packages (``tools/dtb_worlds.py``) with the EFE
task inventory into compiled task entries. The success verdict is
``dtb_all_goals`` — ``EfeDtbTask`` owns the ordered grasp-box latch and
publishes ``info['dtb_goals_reached']``; goals are stored in the meta as
``x y z qx qy qz qw`` and normalised here to ``pos + wxyz`` for the
runtime corner test. ``max_episode_steps`` is the task's own cap from its
instruction (360,000 control steps at 60 Hz), not the suite default.

Usage::

    uv run python tools/compile_efe_suite.py dextoolbench --resolve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "assets" / "dextoolbench" / "worlds"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "dextoolbench"

ROBOT = "kuka_iiwa_sharpa"
#: The instruction's own episode cap ("360,000 control steps (100 min)").
MAX_STEPS = 360_000


class DtbBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = DtbBindingError  # compact alias used throughout the builders


def _pose_to_pos_wxyz(pose: list[float]) -> list[float]:
    """instance.json pose (x y z qx qy qz qw) -> [x, y, z, qw, qx, qy, qz]."""
    if len(pose) != 7:
        raise Reason(f"pose 长度 {len(pose)} != 7: {pose!r}")
    return [
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(pose[6]),
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
    ]


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the DexToolBench suite to the compile knowledge base."""
    if suite != "dextoolbench":
        raise Reason(f"dtb_resolver 只服务 dextoolbench，收到 {suite!r}")
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/dtb_worlds.py --all）")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not (
                TASKS / f"{name}-privileged" / "environment" / "instance.json"
            ).is_file():
                raise Reason("EFE instance.json 缺失（先跑 tools/fetch_dtb_assets.py）")
            goals = [_pose_to_pos_wxyz(g) for g in meta["goals"]]
            world_rel = f"assets/dextoolbench/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": meta.get("text", ""),
                "robot": ROBOT,
                "scene": "efe_world",
                "task_type": "efe_dtb",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/dextoolbench/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/dextoolbench/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/dextoolbench/worlds/{name}.meta.json",
                },
                "dtb": {
                    "goals": goals,
                    "grasp_box_dims_m": [float(v) for v in meta["grasp_box_dims_m"]],
                    "grasp_box_scale": float(meta["grasp_box_scale"]),
                    "corner_tolerance_m": float(meta["corner_tolerance_m"]),
                    "object_body": meta.get("object_body", "object"),
                },
                "success": {"type": "dtb_all_goals", "params": {"target": len(goals)}},
                "max_episode_steps": MAX_STEPS,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据逐字转写 instruction.md（{len(goals)} 个有序 goal，"
                        f"grasp box {meta['grasp_box_dims_m']}x{meta['grasp_box_scale']}，"
                        f"角点容差 {meta['corner_tolerance_m'] * 100:.2f} cm）；"
                        f"场景按 SimToolReal mujoco_sim.py::_init_scene 组合（menagerie kuka + Sharpa HA4 + 桌面 + 工具 CoACD）"
                    ),
                },
                "assets_task_relevant": [
                    {
                        "name": "world",
                        "shape_type": "mjcf",
                        "mjcf_path": world_rel,
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "static": True,
                    }
                ],
            }
        except Reason as exc:
            pending.append(name)
            pending_reasons[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
            pending.append(name)
            pending_reasons[name] = f"{type(exc).__name__}: {exc}"
    return {
        "suite": suite,
        "robot": ROBOT,
        "scene": "efe_world",
        "source": {
            "benchmark": "DexToolBench",
            "repo": "tylerlum/simtoolreal",
            "ref": "313d5ae",
            "simulator": "mujoco",
            "robot": "kuka-iiwa14+sharpa-ha4",
            "note": "场景组合唯一事实来源为 SimToolReal deployment/mujoco/mujoco_sim.py::_init_scene（镜像 @313d5ae）",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
