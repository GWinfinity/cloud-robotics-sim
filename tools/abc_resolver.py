#!/usr/bin/env python3
"""Resolve the ABC Sim EFE suite to the compile knowledge base.

Companion to ``tools/abc_worlds.py`` (world packages) and ``tools/abc_probe.py``
(frozen-instance constants): for every task this resolver merges

* the world-pack meta (joints with qpos addresses, body poses),
* the upstream probe dump (evaluator geometry constants, model card,
  randomization record), and
* the family's verbatim success semantics (``abc_*`` conditions in
  ``cloud_robotics_sim.core.success``),

into the compiled task entries ``compile_efe_suite.py`` writes. The frozen
instance makes every upstream geometry constant a fixed number, so each
success predicate is pinned with literal values — no engine introspection at
runtime.

Success block convention: the EFE suite grades with a 30-consecutive-control-
step hold on top of upstream's single-step evaluators, so non-tray families
publish ``success: {type: abc_hold, steps: 30}`` and carry the verbatim
per-step predicate in ``abc.step_check`` (evaluated by ``EfeAbcTask`` into
``info['abc_success_run']``). ball-tray's countdown verdict and the drawer
sequence latch are task-owned (``abc_tray_complete`` /
``abc_drawer_completed``).

Usage::

    uv run python tools/compile_efe_suite.py abc-sim
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_EFE = ROOT / "data" / "efe"
WORLDS = ROOT / "assets" / "abc-sim" / "worlds"
PROBE_PATH = DATA_EFE / "compile" / "abc-sim.probe.json"

MAX_STEPS = 9000
ROBOT = "i2rt_yam_bimanual"

#: Upstream runtime constants (task_runtime/ball_tray.py) — verbatim.
_TRAY_PULSE_INTERVAL_S = 0.5
_TRAY_PULSE_DURATION_S = 0.1
_TRAY_FORCE_RANGE_N = (0.024, 0.034)
_TRAY_GRACE_S = 0.1
_TRAY_RUN_LENGTH = (3, 5)
_FINGER_BODIES = {
    "left": ("left_link_left_finger", "left_link_right_finger"),
    "right": ("right_link_right_finger", "right_link_left_finger"),
}


class AbcBindingError(Exception):
    """A task goes to ``pending_tasks`` with this human-readable reason."""


Reason = AbcBindingError  # compact alias used throughout the builders


def _load_probe() -> dict[str, Any]:
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


def _family(task: str) -> str:
    match = re.fullmatch(r"abc-sim-(.+)-i\d+", task)
    if match is None:
        raise Reason(f"任务名不解析: {task!r}")
    return match.group(1)


def _joint_map(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Joint name -> {qposadr, qveladr, range} from the world-pack meta."""
    out = {}
    for joint in meta.get("joints", []):
        name = joint.get("name")
        if not name:
            continue
        record = {
            "qposadr": int(joint["qposadr"]),
            "qveladr": int(joint.get("qveladr", 0)),
        }
        if joint.get("range"):
            try:
                record["range"] = [float(v) for v in str(joint["range"]).split()]
            except ValueError:
                pass
        out[name] = record
    return out


def _body_of(
    joints: dict[str, dict[str, Any]], meta: dict[str, Any], joint_name: str
) -> str:
    for joint in meta.get("joints", []):
        if joint.get("name") == joint_name:
            return str(joint.get("body") or joint_name.removesuffix("_joint"))
    return joint_name.removesuffix("_joint")


def _norm3(values: Any, what: str) -> list[float]:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise Reason(f"{what} 不是数值序列: {values!r}") from exc
    if len(out) != 3:
        raise Reason(f"{what} 长度 {len(out)} != 3")
    return out


def _control_block(probe: dict[str, Any]) -> dict[str, Any]:
    card = probe["model_card"]
    ctrl_by_joint = {a["joint"]: a["ctrlrange"] for a in card["actuators"]}
    joints = []
    for entry in card["action_joints"]:
        joints.append(
            {
                "name": entry["joint"],
                "ctrlrange": ctrl_by_joint.get(entry["joint"], entry["range"]),
                "range": entry["range"],
            }
        )
    return {"joints": joints, "control_dt": 0.034}


# ---------------------------------------------------------------------------
# Family builders: (task, probe, meta) -> (success spec, abc block extras)
# ---------------------------------------------------------------------------


def _b_count(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    m = p["randomization"]["metadata"]
    metrics = p["nop_metrics"]
    eligible = set(m.get("eligible_objects") or [])
    objects = [
        {"name": o["name"], "eligible": o["name"] in eligible}
        for o in m.get("objects", [])
    ]
    spec = {
        "type": "abc_count_box",
        "params": {
            "objects": objects,
            "target_count": int(m["target_count"]),
            "x_bounds": [float(v) for v in metrics["box_x_bounds"]],
            "y_bounds": [float(v) for v in metrics["box_y_bounds"]],
            "floor_z": float(metrics["box_floor_z"]),
            "ceiling_z": float(metrics["box_ceiling_z"]),
        },
    }
    return spec, {}


def _b_throw(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    names = metrics["bottle_names"]
    offsets = metrics["bottle_center_offsets"]
    reaches = metrics["bottle_reaches"]
    spec = {
        "type": "abc_bottles_throw",
        "params": {
            "objects": [
                {"name": n, "offset": _norm3(o, "offset"), "reach": float(r)}
                for n, o, r in zip(names, offsets, reaches)
            ],
            "bin": "bin",
            "apothem": float(metrics["bin_apothem"]),
            "floor_z": float(metrics["bin_floor_z"]),
            "top_z": float(metrics["bin_top_z"]),
            "success_count": int(metrics["success_count"]),
        },
    }
    return spec, {}


def _b_put(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    attrs = p["evaluator_attrs"]
    names = metrics["bottle_names"]
    offsets = metrics["bottle_center_offsets"]
    spec = {
        "type": "abc_bottles_put",
        "params": {
            "objects": [
                {"name": n, "offset": _norm3(o, "offset")}
                for n, o in zip(names, offsets)
            ],
            "bin": "bin_container",
            "cone": {
                "bottom_y": float(attrs["bin_bottom_y"]),
                "top_y": float(attrs["bin_top_y"]),
                "radius_bottom": float(attrs["bin_radius_bottom"]),
                "radius_top": float(attrs["bin_radius_top"]),
            },
        },
    }
    return spec, {}


def _b_conveyor(
    p: dict[str, Any], meta: dict[str, Any], joints: dict[str, dict[str, Any]]
) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    m = p["randomization"]["metadata"]
    spec = {
        "type": "abc_in_local_box",
        "params": {
            "objects": [str(n) for n in metrics["object_names"]],
            "frame": "",
            "bounds": {
                "x": [float(v) for v in metrics["bin_x_bounds"]],
                "y": [float(v) for v in metrics["bin_y_bounds"]],
                "z": [
                    float(metrics["bin_floor_z"]) - 0.01,
                    float(metrics["bin_rim_z"]),
                ],
            },
        },
    }
    extras = {
        "conveyor": {
            "objects": [f"conveyor_object_{i}" for i in (1, 2, 3)],
            "belt_geom": "conveyor_belt_surface",
            "belt_speed_y": float(m.get("belt_speed_y", -0.20)),
        }
    }
    return spec, extras


def _b_ball_tray(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    m = p["randomization"]["metadata"]
    attrs = p["evaluator_attrs"]
    spec = {
        "type": "abc_ball_tray",
        "params": {
            "ball": "ball_tray_ball",
            "tray": "ball_tray",
            "usable_half_extents": [float(v) for v in attrs["usable_half_extents"]],
            "dropped_local_z": float(attrs["dropped_local_z_threshold_m"]),
            "require_complete": True,
        },
    }
    extras = {
        "tray": {
            "ball": "ball_tray_ball",
            "tray": "ball_tray",
            "duration_s": int(m.get("balance_duration_s", 12)),
            "grace_s": _TRAY_GRACE_S,
            "usable_half_extents": [float(v) for v in attrs["usable_half_extents"]],
            "dropped_local_z": float(attrs["dropped_local_z_threshold_m"]),
            "left_handle_geom": "ball_tray_left_handle_collision",
            "right_handle_geom": "ball_tray_right_handle_collision",
            "finger_bodies": {
                side: list(bodies) for side, bodies in _FINGER_BODIES.items()
            },
            "pulse_interval_s": _TRAY_PULSE_INTERVAL_S,
            "pulse_duration_s": _TRAY_PULSE_DURATION_S,
            "force_range_n": list(_TRAY_FORCE_RANGE_N),
            "run_length_min": _TRAY_RUN_LENGTH[0],
            "run_length_max": _TRAY_RUN_LENGTH[1],
        }
    }
    return spec, extras


def _b_put_relative(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    m = p["randomization"]["metadata"]
    offset = _norm3(m["direction_offset_xy"] + [0.0], "direction_offset_xy")[:2]
    norm = float(sum(v * v for v in offset) ** 0.5)
    if norm <= 0.0:
        raise Reason("direction_offset_xy 为零向量")
    axis = [round(offset[0] / norm, 6), round(offset[1] / norm, 6)]
    spawn_z = None
    states = p["randomization"].get("object_states") or {}
    state = states.get(f"{m['mover_object']}_joint")
    if isinstance(state, dict) and "pos" in state:
        spawn_z = round(float(state["pos"][2]), 6)
    spec = {
        "type": "abc_put_relative",
        "params": {
            "mover": str(m["mover_object"]),
            "reference": str(m["reference_object"]),
            "axis": axis,
            "spawn_z": spawn_z,
        },
    }
    return spec, {}


def _b_grab_clutter(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    attrs = p["evaluator_attrs"]
    target_body = _body_of({}, meta, "target_object_joint")
    box_body = _body_of({}, meta, "grab_clutter_target_box_joint")
    spec = {
        "type": "abc_in_local_box",
        "params": {
            "objects": [target_body],
            "frame": box_body,
            "local_center": _norm3(attrs["goal_local_center"], "goal_local_center"),
            "half_size": _norm3(attrs["goal_half_size"], "goal_half_size"),
        },
    }
    return spec, {}


def _b_inhand(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    m = p["randomization"]["metadata"]
    attrs = p["evaluator_attrs"]
    side = m.get("side")
    if side not in ("left", "right"):
        raise Reason(f"side 缺失/非法: {side!r}")
    obj = _body_of({}, meta, "task_object_joint")
    spec = {
        "type": "abc_inhand_transfer",
        "params": {
            "object": obj,
            "start_sign": 1.0 if side == "left" else -1.0,
            "table_x_bounds": [float(v) for v in attrs["table_x_bounds"]],
            "table_y_bounds": [float(v) for v in attrs["table_y_bounds"]],
            "table_z": float(attrs["table_z"]),
        },
    }
    return spec, {}


def _b_pour(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    attrs = p["evaluator_attrs"]
    cup = _body_of({}, meta, str(attrs["target_joint_name"]))
    spec = {
        "type": "abc_pour",
        "params": {
            "beads": [str(n) for n in metrics["bead_names"]],
            "cup": cup,
            "radius": float(attrs["container_radius"]),
            "floor_z": float(attrs["container_floor_z"]),
            "rim_z": float(attrs["container_rim_z"]),
            "success_count": int(metrics["success_count"]),
        },
    }
    return spec, {}


def _b_hang_mug(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    attrs = p["evaluator_attrs"]
    mugs = [str(n) for n in metrics["mug_names"]]
    fingers = [b for bodies in _FINGER_BODIES.values() for b in bodies]
    spec = {
        "type": "all",
        "params": {
            "conditions": [
                {
                    "type": "abc_mug_tree",
                    "params": {
                        "mugs": mugs,
                        "axis_xy": [float(v) for v in attrs["axis_xy"]],
                        "hang_radius": float(metrics["hang_radius_m"]),
                        "z_band": [float(v) for v in metrics["hang_z_band"]],
                    },
                },
                {
                    "type": "abc_not_touching",
                    "params": {
                        "subjects": mugs,
                        "obstacles": ["table_plane", *fingers],
                    },
                },
            ]
        },
    }
    return spec, {}


def _b_turn_mug(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    bodies = meta.get("bodies") or {}
    mugs = []
    for name in metrics["mug_names"]:
        world = (bodies.get(name) or {}).get("world")
        if not world:
            raise Reason(f"mug {name} 无冻结位姿")
        mugs.append({"name": name, "rest_z": round(float(world[2]), 6)})
    spec = {"type": "abc_mugs_upright", "params": {"mugs": mugs}}
    return spec, {}


def _b_sweep(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]
    names = [str(n) for n in metrics["trash_names"]]
    spec = {
        "type": "abc_sweep",
        "params": {
            "scraps": names,
            "bin": "bin",
            "bin_radius": float(metrics["bin_radius"]),
            "bin_radial_margin": 0.005,
            "bin_bottom_y": float(metrics["bin_bottom_y"]),
            "bin_top_y": float(metrics["bin_top_y"]),
            "bin_height_margin": 0.005,
        },
    }
    extras = {
        "sweep": {"scraps": names, "active_z_min": float(metrics["active_z_min"])}
    }
    return spec, extras


def _b_dish_rack(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    metrics = p["nop_metrics"]

    def unwrap(value: Any) -> Any:
        # DishrackEvaluator reports per-row lists for scalar metrics.
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        return value

    fingers = [b for bodies in _FINGER_BODIES.values() for b in bodies]
    spec = {
        "type": "abc_dishrack",
        "params": {
            "plates": [str(n) for n in metrics["plate_names"]],
            "rack": "dishrack",
            "rack_half_extents": [
                float(v) for v in unwrap(metrics["rack_half_extents_xy"])
            ],
            "margin": float(unwrap(metrics["rack_margin_m"])),
            "height_band": [float(v) for v in unwrap(metrics["rack_height_band_m"])],
            "upright_normal_z_max": float(unwrap(metrics["upright_normal_z_max"])),
            "gripper_bodies": fingers,
        },
    }
    return spec, {}


def _b_chess(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    attrs = p["evaluator_attrs"]
    m = p["randomization"]["metadata"]
    joints = attrs["chess_joints"]
    goals = attrs["chess_goals_board"]
    classes = attrs["chess_classes"]
    board = _body_of({}, meta, "chessboard")
    non_joints = [str(j) for j in (m.get("non_target_pieces") or [])]
    non_goals = attrs.get("chess_non_target_goals") or []
    non_targets = [
        {"name": name, "goal": _norm3(goal, "non_target goal")}
        for name, goal in zip(non_joints, non_goals)
    ]
    spec = {
        "type": "abc_chess",
        "params": {
            "board": board,
            "normal_local": _norm3(attrs["chess_normal_local"], "normal_local"),
            "targets": [
                {"name": name, "goal": _norm3(goal, "goal")}
                for name, goal in zip(joints, goals)
            ],
            "classes": [str(c) for c in classes],
            "non_targets": non_targets,
            "position_tolerance": float(attrs["position_tolerance_m"]),
            "height_tolerance": float(attrs["height_tolerance_m"]),
            "upright_z_min": float(attrs["upright_z_min"]),
        },
    }
    return spec, {}


def _b_spell(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict, dict]:
    attrs = p["evaluator_attrs"]
    metrics = p["nop_metrics"]
    bodies = meta.get("bodies") or {}
    letters = []
    rest_z = []
    for letter in metrics["word"]:
        name = f"block_{letter}_jnt"
        body = _body_of({}, meta, name)
        world = (bodies.get(body) or {}).get("world")
        if not world:
            raise Reason(f"字母方块 {body} 无冻结位姿")
        letters.append(body)
        rest_z.append(round(float(world[2]), 6))
    spec = {
        "type": "abc_spell",
        "params": {
            "letters": letters,
            "gap_min": float(attrs["gap_min_m"]),
            "gap_max": float(attrs["gap_max_m"]),
            "row_dx_max": float(attrs["row_dx_max_m"]),
            "rest_z": rest_z,
            "z_tolerance": float(attrs["z_tolerance_m"]),
            "upright_dot_min": float(attrs["upright_dot_min"]),
        },
    }
    return spec, {}


def _b_lego(
    p: dict[str, Any], meta: dict[str, Any], joints: dict[str, dict[str, Any]]
) -> tuple[dict, dict]:
    attrs = p["evaluator_attrs"]
    m = p["randomization"]["metadata"]
    active = set(m.get("active_object_joints") or [])
    conditions = []
    for color, (center, half) in sorted(attrs["goal_regions"].items()):
        pattern = re.compile(rf"{re.escape(color)}_lego_(\d+)_joint")
        names = []
        for joint_name in joints:
            match = pattern.fullmatch(joint_name)
            if match is None:
                continue
            if active and joint_name not in active:
                continue
            names.append(_body_of({}, meta, joint_name))
        if not names:
            raise Reason(f"lego 颜色 {color} 无激活方块")
        conditions.append(
            {
                "type": "abc_in_local_box",
                "params": {
                    "objects": names,
                    "frame": _body_of({}, meta, f"{color}_lego_sorting_bin_joint"),
                    "local_center": _norm3(center, "goal center"),
                    "half_size": _norm3(half, "goal half"),
                },
            }
        )
    spec = {"type": "all", "params": {"conditions": conditions}}
    return spec, {}


def _b_nuts_bolts(
    p: dict[str, Any], meta: dict[str, Any], joints: dict[str, dict[str, Any]]
) -> tuple[dict, dict]:
    attrs = p["evaluator_attrs"]
    m = p["randomization"]["metadata"]
    active = set(m.get("active_object_joints") or [])

    def names(pattern_str: str) -> list[str]:
        pattern = re.compile(pattern_str)
        out = []
        for joint_name in joints:
            if pattern.fullmatch(joint_name) is None:
                continue
            if active and joint_name not in active:
                continue
            out.append(_body_of({}, meta, joint_name))
        return out

    nut_names = names(r"nut_(\d+)_joint")
    bolt_names = names(r"bolt_(\d+)_joint")
    if not nut_names or not bolt_names:
        raise Reason(
            f"nuts/bolts 激活集合空: nuts={len(nut_names)} bolts={len(bolt_names)}"
        )
    spec = {
        "type": "all",
        "params": {
            "conditions": [
                {
                    "type": "abc_in_local_box",
                    "params": {
                        "objects": nut_names,
                        "frame": _body_of({}, meta, "nuts_bin_joint"),
                        "local_center": _norm3(
                            attrs["nuts_goal_local_center"], "nuts center"
                        ),
                        "half_size": _norm3(attrs["nuts_goal_size"], "nuts half"),
                    },
                },
                {
                    "type": "abc_in_local_box",
                    "params": {
                        "objects": bolt_names,
                        "frame": _body_of({}, meta, "bolts_bin_joint"),
                        "local_center": _norm3(
                            attrs["bolts_goal_local_center"], "bolts center"
                        ),
                        "half_size": _norm3(attrs["bolts_goal_size"], "bolts half"),
                    },
                },
            ]
        },
    }
    return spec, {}


def _b_drawer(p: dict[str, Any], meta: dict[str, Any]) -> tuple[dict | None, dict]:
    m = p["randomization"]["metadata"]
    attrs = p["evaluator_attrs"]
    sequence = [str(t["body"]) for t in m.get("target_sequence", [])]
    if not sequence:
        target = m.get("target_body")
        if not target:
            raise Reason("target_sequence/target_body 均缺失")
        sequence = [str(target)]
    sites = meta.get("sites") or {}
    site = sites.get("drawer_search_goal_region")
    if not site:
        raise Reason("drawer_search_goal_region 站点缺失")
    spec = {"type": "abc_drawer_sequence", "params": {"target_count": len(sequence)}}
    extras = {
        "drawer": {
            "targets": sequence,
            "bin": str(site["body"]),
            "local_center": _norm3(attrs["goal_local_center"], "goal center"),
            "half_size": _norm3(attrs["goal_half_size"], "goal half"),
        }
    }
    return spec, extras


_BUILDERS = {
    "count": _b_count,
    "count_one": _b_count,
    "count_two": _b_count,
    "count_three": _b_count,
    "count_four": _b_count,
    "throw_bottles": _b_throw,
    "put_bottles": _b_put,
    "conveyor_pick": _b_conveyor,
    "ball_tray": _b_ball_tray,
    "put_relative": _b_put_relative,
    "grab_clutter": _b_grab_clutter,
    "inhand_transfer": _b_inhand,
    "pouring": _b_pour,
    "hang_mug": _b_hang_mug,
    "turn_mug": _b_turn_mug,
    "sweep": _b_sweep,
    "dish_rack": _b_dish_rack,
    "chess": _b_chess,
    "spell_cat": _b_spell,
    "spell_dog": _b_spell,
    "spell_fish": _b_spell,
    "spell_bair": _b_spell,
    "spell_xdof": _b_spell,
    "spell_abc": _b_spell,
    "spell_yam": _b_spell,
    "spell_agi": _b_spell,
    "lego": _b_lego,
    "nuts_bolts": _b_nuts_bolts,
    "drawer_search": _b_drawer,
}


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve the ABC Sim suite to the compile knowledge base."""
    if suite != "abc-sim":
        raise Reason(f"abc_resolver 只服务 abc-sim，收到 {suite!r}")
    probe = _load_probe()
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            p = probe.get(name)
            if not p or "error" in p:
                raise Reason(
                    f"探针缺失/失败: {(p or {}).get('error', '未运行 abc_probe')}"
                )
            if not p.get("instance_exact"):
                raise Reason("冻结实例状态哈希未逐位复现")
            meta_path = WORLDS / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/abc_worlds.py --all）")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            joints = _joint_map(meta)
            family = _family(name).replace("-", "_")
            builder = _BUILDERS.get(family)
            if builder is None:
                raise Reason(f"家族 {family!r} 无 builder")
            try:
                step_check, extras = builder(p, meta, joints)
            except TypeError:
                step_check, extras = builder(p, meta)  # type: ignore[misc]

            seed = (
                p["randomization"].get("seed")
                or p["randomization"].get("metadata", {}).get("seed")
                or 0
            )
            abc: dict[str, Any] = {
                "family": family,
                "rule": p.get("rule", ""),
                "hold_steps": 30,
                "step_check": step_check,
                "control": _control_block(p),
                "joints": joints,
                "randomization_seed": int(seed),
            }
            abc.update(extras)

            # success 块：hold 家族走 abc_hold；ball_tray/drawer 由 task 发布裁决
            if "tray" in extras:
                success = step_check
            elif "drawer" in extras:
                success = step_check
            else:
                success = {"type": "abc_hold", "params": {"steps": 30}}

            world_rel = f"assets/abc-sim/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": (p["randomization"].get("metadata") or {}).get("prompt") or "",
                "robot": ROBOT,
                "scene": "efe_world",
                "task_type": "efe_abc",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/abc-sim/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/abc-sim/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/abc-sim/worlds/{name}.meta.json",
                },
                "abc": abc,
                "success": success,
                "max_episode_steps": MAX_STEPS,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据逐字转写上游 {p['upstream_task']} evaluator + EFE 三处修订"
                        f"（30 步 hold / 挂杯释放检查 / 托盘倒计时）；"
                        f"instance_sha={str(p.get('instance_sha', '?'))[:12]}，"
                        f"探针逐位复现={bool(p.get('instance_exact'))}"
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
            "benchmark": "ABC Sim",
            "repo": "amazon-far/abc",
            "ref": "d0832d12651d1b260a652861a14648dc5f3660c7",
            "simulator": "mujoco",
            "robot": "i2rt-yam-bimanual",
            "note": "判据常量的唯一事实来源是 data/efe/compile/abc-sim.probe.json（上游代码自测导出，29/29 状态哈希逐位复现）",
        },
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }
