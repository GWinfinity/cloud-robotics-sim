#!/usr/bin/env python3
"""Resolve the RoboWits EFE suite to the compile knowledge base.

Only the three BlenderKit-free families (dominos / stand_pages /
stack_cubes) resolve; the other twelve need the non-redistributable
BlenderKit models (images/robowits/NOTICE) and stay pending with that
reason. The success verdict is ``rw_success`` — ``EfeRwTask`` ports the
family's gs_gym ``_check_success``, the graded-object release clause and
the rcb_robowits settle window (SETTLE_STEPS hold, then CHECK_STEPS with
CHECK_STRIDE checks and the REST_M rest bound). ``max_episode_steps`` is
the runaway guard (rcb_robowits MAX_STEPS, 200,000 control steps).

Usage::

    uv run python tools/compile_efe_suite.py robowits --resolve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "assets" / "robowits" / "worlds"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "robowits"

BLENDERKIT_REASON = (
    "该家族需要非再分发 BlenderKit 模型（images/robowits/NOTICE，需个人 key 或加密研究包）；"
    "本镜像只含无 BlenderKit 的三个家族（dominos/stand_pages/stack_cubes）"
)

#: Meta fields inlined into the rw block of the compiled configs.
_RW_KEYS = (
    "family",
    "text",
    "objects",
    "graded",
    "settle",
    "gripper_open",
    "table_z",
    "table_bounds",
    "ee_links",
    "robot_joints",
)


class RwBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = RwBindingError


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the RoboWits suite to the compile knowledge base."""
    if suite != "robowits":
        raise Reason(f"rw_resolver 只服务 robowits，收到 {suite!r}")
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            family = (
                name.removesuffix("-i00").removeprefix("robowits-").replace("-", "_")
            )
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason(
                    BLENDERKIT_REASON
                    if family not in ("dominos", "stand_pages", "stack_cubes")
                    else "世界包缺失（先跑 tools/rw_worlds.py --all）"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not (
                TASKS / f"{name}-privileged" / "environment" / "instance.json"
            ).is_file():
                raise Reason("EFE instance.json 缺失（先跑 tools/fetch_rw_assets.py）")
            world_rel = f"assets/robowits/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": meta.get("text", ""),
                "robot": "marvin_bimanual",
                "scene": "efe_world",
                "task_type": "efe_rw",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/robowits/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/robowits/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/robowits/worlds/{name}.meta.json",
                },
                "rw": {k: meta.get(k) for k in _RW_KEYS if meta.get(k) is not None},
                "success": {"type": "rw_success", "params": {}},
                "max_episode_steps": 200_000,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据逐字移植 gs_gym {family}._check_success + rcb_robowits settle 窗口"
                        f"（{meta['settle']['settle_steps']}+{meta['settle']['check_steps']} 步，"
                        f"每 {meta['settle']['check_stride']} 步检，静止 {meta['settle']['rest_m'] * 1000:.0f} mm）；"
                        f"世界=Marvin URDF(obj 转换)+工作台盒({0.85}x{1.5}x{0.76})+冻结对象；"
                        "hold 计数闭合环近似 open-loop settle replay"
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
        "robot": "marvin_bimanual",
        "scene": "efe_world",
        "source": {
            "benchmark": "RoboWits",
            "repo": "UMass-Embodied-AGI/RoboWits",
            "ref": "9cc30aedbbebea86b975c0d66ea9ac227ca38535",
            "simulator": "genesis",
            "note": "场景唯一事实来源为 RoboWits gs_gym 任务类 + EFE rcb_robowits；"
            "资产=HF RoboWits_assets@b1b49abb（Apache-2.0）",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
