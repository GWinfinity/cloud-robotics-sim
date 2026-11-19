#!/usr/bin/env python3
"""Resolve the DexJoCo EFE suite to the compile knowledge base.

Merges the replayed world packages (``tools/djx_worlds.py``) with the EFE
inventory. Six of the eleven families port verbatim from pose/joint/contact
readables (see :class:`EfeDjxTask`); the other five stay ``pending`` with
the engine-quantity reason recorded: ``pinch_tongs`` (tongs_pos sensor
frame mapping unverified), ``hammer_nail`` (pre-impact hammer velocity
needs a qvel getter the world entity does not expose), ``bimanual_hanoi``
(disk-tower reconstruction geometry), ``bimanual_unlock_ipad``
(digit-sequence + screen-material state machine) and ``bimanual_photograph``
(static display-geom world poses). The success verdict is ``djx_success``
(the episode ends at the first success, as in DexJoCo's own loop).

Usage::

    uv run python tools/compile_efe_suite.py dexjoco --resolve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "assets" / "dexjoco" / "worlds"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "dexjoco"

#: Families whose ``_compute_success`` ports verbatim (see EfeDjxTask).
PORTED = {
    "water_plant",
    "pick_bucket",
    "fold_glasses",
    "bimanual_microwave_cook",
    "bimanual_assembly",
    "click_mouse",
    "pinch_tongs",
    "hammer_nail",
    "bimanual_hanoi",
    "bimanual_unlock_ipad",
    "bimanual_photograph",
}
PENDING_REASONS: dict[str, str] = {}
ROBOT_BY_FAMILY_PREFIX = "franka_allegro_bimanual"
MAX_STEPS = 180_000  # 3600 s agent budget at 50 Hz control


class DjxBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = DjxBindingError  # compact alias used throughout the builders


def _family(task: str) -> str:
    stem = task.removesuffix("-i00")
    return stem.removeprefix("dexjoco-").replace("-", "_")


def _djx_block(family: str, meta: dict) -> dict[str, Any]:
    """Per-family constants for the EfeDjxTask success port."""
    block: dict[str, Any] = {"family": family}
    if family == "pick_bucket":
        sites = meta.get("sites") or {}
        bottoms = [f"bucket_ref_{i}" for i in (0, 2, 4, 6)]
        block["object_body"] = "boxed_food_0"
        block["bucket_bottom_sites"] = bottoms
        block["bucket_bottom_z0"] = float(sites[bottoms[0]]["world"][2])
    elif family == "click_mouse":
        geoms = meta.get("geoms") or {}
        block["mousepad_radius"] = float(geoms["mousepad"]["size"][0])
    elif family == "pinch_tongs":
        # 上游 table_z=0.92 硬编码 + _LIFT_HEIGHT 0.1（env 构造器逐字）。
        block["lift_z"] = 1.02
    elif family == "hammer_nail":
        attrs = meta.get("attrs") or {}
        block["success_depth"] = float(attrs.get("_success_depth", 0.04))
        block["max_insert_depth"] = float(attrs.get("_nail_max_depth", 0.0726))
        block["impact_insert_step"] = float(attrs.get("_impact_insert_step", 0.008))
        block["impact_vel_threshold"] = float(attrs.get("_impact_vel_threshold", 0.02))
    elif family == "bimanual_hanoi":
        attrs = meta.get("attrs") or {}
        sites = meta.get("sites") or {}
        geoms = meta.get("geoms") or {}
        block["post_assignment_tol"] = float(attrs.get("_post_assignment_tol", 0.04))
        block["placement_height_tol"] = float(attrs.get("_placement_height_tol", 0.012))
        block["illegal_trigger_target"] = int(attrs.get("_illegal_trigger_target", 10))
        upper = geoms["hanoi_base_upper_collision"]
        block["base_upper_top_local_z"] = float(upper["local"][2]) + float(
            upper["size"][2]
        )
        halves = {}
        for disk in (
            "hanoi_disk_large",
            "hanoi_disk_medium",
            "hanoi_disk_small",
        ):
            top = float(sites[f"{disk}_top_site"]["local"][2])
            bottom = float(sites[f"{disk}_bottom_site"]["local"][2])
            halves[disk] = 0.5 * (top - bottom)
        block["disk_half_height"] = halves
        block["success_target"] = int(attrs.get("_success_trigger_target", 20))
    elif family == "bimanual_unlock_ipad":
        block["unlock_sequence"] = [1, 2, 3]
    return block


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the DexJoCo suite to the compile knowledge base."""
    if suite != "dexjoco":
        raise Reason(f"djx_resolver 只服务 dexjoco，收到 {suite!r}")
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/djx_worlds.py --all）")
            if not (
                TASKS / f"{name}-privileged" / "environment" / "instance.npz"
            ).is_file():
                raise Reason("EFE instance.npz 缺失（先跑 tools/fetch_djx_assets.py）")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            family = meta["family"]
            if family not in PORTED:
                raise Reason(PENDING_REASONS.get(family, f"家族 {family} 未移植"))
            robot = (
                ROBOT_BY_FAMILY_PREFIX
                if family.startswith("bimanual")
                else "franka_allegro"
            )
            world_rel = f"assets/dexjoco/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": "",
                "robot": robot,
                "scene": "efe_world",
                "task_type": "efe_djx",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/dexjoco/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/dexjoco/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/dexjoco/worlds/{name}.meta.json",
                },
                "djx": _djx_block(family, meta),
                "success": {"type": "djx_success"},
                "max_episode_steps": MAX_STEPS,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据逐字转写上游 {family}._compute_success（首次成功即终局，无 hold）；"
                        f"世界包=env XML + {meta['model_arrays_applied']} 个模型数组回放；"
                        f"instance_sha={meta['instance_sha']}"
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
        "robot": "franka_allegro(+bimanual)",
        "scene": "efe_world",
        "source": {
            "benchmark": "DexJoCo",
            "repo": "brave-eai/dexjoco",
            "ref": "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
            "simulator": "mujoco",
            "robot": "franka-allegro",
            "note": "世界=env XML + model__ 数组回放（restorer 语义逐字）；判据=EfeDjxTask 逐字移植",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
