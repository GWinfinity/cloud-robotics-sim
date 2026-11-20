#!/usr/bin/env python3
"""Resolve the HumanoidBench EFE suite to the compile knowledge base.

Merges the composed world packages (``tools/hb_worlds.py``) with the EFE task
inventory into compiled task entries. The success verdict is ``hb_success`` —
HumanoidBench's own rule (the episode's summed dense reward reaches the task's
success bar; the episode ends on the task's ported ``get_terminated``).
``EfeHbTask`` owns the ten ported reward functions and the accumulation; the
thin condition forwards ``info['hb']['success']``. The ``hb`` block inlines
the frozen-instance constants (success bar, termination spec, class/module
constants, task state, site table, robot link/mass table) so the compiled
configs are self-contained.

``max_episode_steps`` is the task's own episode cap (1000 control steps at
50 Hz; 500 for cube where the instance caps it).

Usage::

    uv run python tools/compile_efe_suite.py humanoidbench --resolve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "assets" / "humanoidbench" / "worlds"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "humanoidbench"

ROBOT_BY_NAME = {
    "h1": "unitree_h1_embedded",
    "h1hand": "unitree_h1hand_embedded",
}

#: Check-constant fields inlined from the world-pack meta into the hb block.
_HB_KEYS = (
    "task_short",
    "env_id",
    "success_bar",
    "nu",
    "sites",
    "termination",
    "class_consts",
    "module_globals",
    "task_state",
    "room_objects",
    "robot_links",
    "actuated",
    "tool_subtree",
)


class HbBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = HbBindingError  # compact alias used throughout the builders


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the HumanoidBench suite to the compile knowledge base."""
    if suite != "humanoidbench":
        raise Reason(f"hb_resolver 只服务 humanoidbench，收到 {suite!r}")
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/hb_worlds.py --all）")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not (
                TASKS / f"{name}-privileged" / "environment" / "instance.npz"
            ).is_file():
                raise Reason("EFE instance.npz 缺失（先跑 tools/fetch_hb_assets.py）")
            robot = ROBOT_BY_NAME.get(meta.get("robot", ""))
            if robot is None:
                raise Reason(f"未注册的 hb 机器人: {meta.get('robot')!r}")
            max_steps = int(meta.get("max_steps", 1000))
            world_rel = f"assets/humanoidbench/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": meta.get("task_short", ""),
                "robot": robot,
                "scene": "efe_world",
                "task_type": "efe_hb",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/humanoidbench/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/humanoidbench/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/humanoidbench/worlds/{name}.meta.json",
                },
                "hb": {k: meta.get(k) for k in _HB_KEYS if meta.get(k) is not None},
                "success": {"type": "hb_success", "params": {}},
                "max_episode_steps": max_steps,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据=HumanoidBench 规则：累计稠密奖励 >= {meta['success_bar']}"
                        f"（bar 逐字自上游 {meta['task_short']} 类），episode 上限 "
                        f"{max_steps} 控制步 @ 50 Hz；reward/termination 逐字移植 "
                        "rewards.json + envs/*.py（dm_control tolerance 语义完整移植）；"
                        "物理=上游 cb118903 + EFE 修复（box_feet/upstream-fixes）"
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
        "robot": "unitree_h1_embedded",
        "scene": "efe_world",
        "source": {
            "benchmark": "HumanoidBench",
            "repo": "carlosferrazza/humanoid-bench",
            "ref": "cb1189039151c8aadaaa987b442da54383c87fab",
            "simulator": "mujoco+dm_control",
            "note": "场景唯一事实来源为 rcb_hb.Session + EFE image 补丁"
            "（no-torch / upstream-fixes / box_feet，mujoco 3.3.7 + dm_control 1.0.34）",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
