#!/usr/bin/env python3
"""Resolve the MuJoCo Playground EFE suite to the compile knowledge base.

Merges the composed world packages (``tools/pg_worlds.py``) with the EFE task
inventory into compiled task entries. The success verdict is ``pg_success`` —
``EfePgTask`` owns the graders' check formulas (``rcb_quadruped.py`` /
``rcb_aloha.py`` / ``rcb_panda.py``, mirrored under
``data/efe/upstream/pg-src/efe-benchmark``), publishes ``info['pg']`` each
control step, and the thin condition forwards it. The ``pg`` block inlines the
check constants (segments / kicks / gait / termination boxes / sites / peg
axial limits / targets) so the compiled configs are self-contained.

``max_episode_steps`` is the upstream horizon for the quadrupeds (joystick
success only at the horizon; getup/stand end the moment their hold is met)
and the sessions' 3000-control-step cap for the manipulation tasks.

Usage::

    uv run python tools/compile_efe_suite.py mujoco-playground --resolve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "assets" / "playground" / "worlds"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "mujoco-playground"

#: World-embedded runtime robots (registered in runtime/efe.py).
ROBOT_BY_PG_ROBOT = {
    "go1": "unitree_go1",
    "spot": "boston_dynamics_spot",
    "barkour": "google_barkour_vb",
    "aloha": "aloha_2",
    "panda": "franka_panda_embedded",
}

#: Check-constant fields inlined from the world-pack meta into the pg block.
_QUAD_KEYS = (
    "task_kind",
    "horizon",
    "control_dt",
    "constants",
    "commands",
    "kicks",
    "gait",
    "imu_site",
    "root_body",
    "floor_geom",
    "feet",
    "sites",
    "joint_names",
    "ctrlrange",
)
_MANIP_KEYS = (
    "task_kind",
    "actuators",
    "max_steps",
    "object_body",
    "target_pos",
    "target_quat_wxyz",
    "sites",
    "check",
)


class PgBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = PgBindingError  # compact alias used throughout the builders


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the MuJoCo Playground suite to the compile knowledge base."""
    if suite != "mujoco-playground":
        raise Reason(f"pg_resolver 只服务 mujoco-playground，收到 {suite!r}")
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/pg_worlds.py --all）")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not (
                TASKS / f"{name}-privileged" / "environment" / "instance.json"
            ).is_file():
                raise Reason("EFE instance.json 缺失（先跑 tools/fetch_pg_assets.py）")
            kind_group = meta.get("kind_group")
            if kind_group == "quadruped":
                robot_key = meta.get("robot", "")
                max_steps = int(meta["horizon"])
                pg_block = {
                    k: meta.get(k) for k in _QUAD_KEYS if meta.get(k) is not None
                }
                criterion = (
                    f"{meta['task_kind']} 判据逐字转写 rcb_quadruped.py::Session.check"
                    f"（horizon {max_steps} 控制步 @ {meta['control_dt']} s）"
                )
            elif kind_group == "manipulation":
                robot_key = "aloha" if "Aloha" in meta.get("env", "") else "panda"
                max_steps = int(meta.get("max_steps", 3000))
                pg_block = {
                    k: meta.get(k) for k in _MANIP_KEYS if meta.get(k) is not None
                }
                criterion = (
                    f"{meta['env']} 判据逐字转写 "
                    "rcb_aloha.py::Session.check / rcb_panda.py::Session.check"
                    f"（3000 控制步上限 @ 0.02 s）"
                )
            else:
                raise Reason(f"未知 kind_group: {kind_group!r}")
            robot = ROBOT_BY_PG_ROBOT.get(robot_key)
            if robot is None:
                raise Reason(f"未注册的 pg 机器人: {robot_key!r}")
            world_rel = f"assets/playground/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": meta.get("text", ""),
                "robot": robot,
                "scene": "efe_world",
                "task_type": "efe_pg",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/playground/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/playground/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/playground/worlds/{name}.meta.json",
                },
                "pg": pg_block,
                "success": {"type": "pg_success", "params": {}},
                "max_episode_steps": max_steps,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"{criterion}；场景按 rcb_* Session 复刻"
                        "（playground@4057c14 + menagerie@1b86ece，"
                        "增益/关键帧/实例随机化/settle 一致；"
                        "mocap 目标转为静态可视标记，Barkour 推力不重现）"
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
        "robot": "unitree_go1",
        "scene": "efe_world",
        "source": {
            "benchmark": "MuJoCo Playground",
            "repo": "google-deepmind/mujoco_playground",
            "ref": "4057c14",
            "simulator": "mujoco",
            "note": "场景唯一事实来源为 efe-benchmark/rcb_{quadruped,aloha,panda}.py::load_model + reset"
            "（镜像 @4057c14 + menagerie@1b86ece）",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
