#!/usr/bin/env python3
"""Compose Genesis-ready RoboWits world packages from the EFE instances.

RoboWits runs the Tianji Marvin bimanual robot (URDF + glb meshes from the
Apache-2.0 HF bundle) on a work table with primitive puzzle objects. The
probe venv (``.abcvenv``, mujoco 3.6 + trimesh) stages a MuJoCo world per
family: the URDF imported with obj-converted meshes and position actuators
on the 18 joints (7+2 per arm), the work table as a static box
(``TABLE_SIZE`` 0.85 x 1.5 x 0.76 at ``TABLE_POS`` (0.597, 0, 0.38), per
``gs_gym/envs/robowits/robowits.py``), and the frozen objects of the EFE
``instance.json`` scene entry as free boxes. Writes under
``assets/robowits/worlds/``:

* ``<task>.xml`` — the composed world (mesh paths absolute);
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen t = 0 state
  (robot reset joints from ``reset_joint_targets`` + object free poses);
* ``<task>.meta.json`` — family text, object table (size/pos/fixed), the
  settle-window constants (SETTLE_STEPS 30 / CHECK_STEPS 60 /
  CHECK_STRIDE 6 / REST_M 0.002), the graded-object lists, the action
  layout (16 rows -> 18 dofs) and the EE link names.

Only the three BlenderKit-free families are composed (dominos /
stand_pages / stack_cubes); the other twelve need the non-redistributable
BlenderKit models and stay pending.

Usage::

    .abcvenv/Scripts/python.exe tools/rw_worlds.py --all
    .abcvenv/Scripts/python.exe tools/rw_worlds.py --task robowits-dominos-i00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets" / "robowits"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "robowits"
WORLDS = BUNDLE / "worlds"
OBJ_MESH_DIR = BUNDLE / "meshes_obj"

URDF_SRC = BUNDLE / "marvin_bimanual" / "urdf" / "marvin_pika.urdf"

#: gs_gym/envs/robowits/robowits.py + configs/envs/robowits_default.yaml
TABLE_SIZE = (0.85, 1.5, 0.76)
TABLE_POS = (0.597, 0.0, 0.38)
ROBOT_POS = (0.0, 0.0, 1.08)
TABLE_Z = 0.76

#: rcb_robowits.py settle window
SETTLE_STEPS = 30
CHECK_STEPS = 60
CHECK_STRIDE = 6
REST_M = 0.002
GRIPPER_OPEN = 0.1

#: The three BlenderKit-free families (rcb_robowits TASKS + GRADED).
FAMILIES = {
    "dominos": {
        "text": "Push the red block onto the green target area.",
        "rest": ["red block"],
        "release_also": [],
    },
    "stand_pages": {
        "text": "Assemble the two pages using the stabilizing bar to create a standing hinge structure.",
        "rest": ["page A with eyelets", "page B with eyelets"],
        "release_also": ["stabilizing bar"],
    },
    "stack_cubes": {
        "text": "Build a stable two-layer stack so the red cube's center is above the green colored dot.",
        "rest": ["base cube 1", "base cube 2", "apex cube"],
        "release_also": [],
    },
}

ARM_JOINTS = [f"Joint{i}_R" for i in range(1, 8)] + [f"Joint{i}_L" for i in range(1, 8)]
GRIP_JOINTS = ["Joint8_R", "Joint9_R", "Joint8_L", "Joint9_L"]
EE_LINKS = {"right": "Gripper_Tip_R", "left": "Gripper_Tip_L"}
TABLE_BOUNDS = {"x": [0.21, 1.00], "y": [-0.69, 0.69]}


def _euler_to_quat_wxyz(euler: list[float]) -> np.ndarray:
    """Genesis/URDF euler_xyz (extrinsic XYZ) -> wxyz quat."""
    import math

    rx, ry, rz = (math.radians(float(v)) for v in euler)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    # R = Rz @ Ry @ Rx (extrinsic XYZ)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _stage_urdf() -> tuple[Path, dict[str, bytes]]:
    """Convert the glb meshes to obj; return (staged urdf, obj assets)."""
    import trimesh

    OBJ_MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh_dir = BUNDLE / "marvin_bimanual" / "meshes"
    assets: dict[str, bytes] = {}
    for glb in sorted(mesh_dir.glob("*.glb")):
        obj_path = OBJ_MESH_DIR / (glb.stem + ".obj")
        if not obj_path.is_file():
            mesh = trimesh.load(glb, force="mesh")
            obj_path.write_text(mesh.export(file_type="obj"), encoding="utf-8")
        assets[obj_path.name] = obj_path.read_bytes()
    text = URDF_SRC.read_text(encoding="utf-8").replace(".glb", ".obj")
    staged = WORLDS / "_marvin_pika.urdf"
    WORLDS.mkdir(parents=True, exist_ok=True)
    staged.write_text(text, encoding="utf-8")
    return staged, assets


def _add_position_actuators(spec: Any, model: Any) -> None:
    import mujoco

    for i in range(model.njnt):
        name = model.joint(i).name
        if name not in ARM_JOINTS + GRIP_JOINTS:
            continue
        act = spec.add_actuator()
        act.name = name
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(100.0, 10.0)
        lo, hi = (float(v) for v in model.jnt_range[i])
        act.ctrlrange = [lo, hi]
        act.forcerange = [-100.0, 100.0]


def _absolutize_xml(xml: str) -> str:
    import re

    def _resolve(match: Any) -> str:
        attr, ref = match.group(1), match.group(2)
        if "\\" in ref:
            ref = ref.replace("\\", "/")
        if Path(ref).is_file():
            return f'{attr}="{ref}"'
        name = Path(ref).name
        cand = OBJ_MESH_DIR / name
        if cand.is_file():
            return f'{attr}="{cand.as_posix()}"'
        raise SystemExit(f"mesh not found in mirror: {ref}")

    return re.sub(r'(\sfile)="([^"]+)"', _resolve, xml)


def compose_one(task: str) -> dict[str, Any]:
    """Compose one RoboWits world package; return its meta."""
    import mujoco

    family = task.removesuffix("-i00").removeprefix("robowits-").replace("-", "_")
    if family not in FAMILIES:
        raise SystemExit(f"{task}: family {family!r} not in the BlenderKit-free set")
    inst = json.loads(
        (TASKS / f"{task}-privileged" / "environment" / "instance.json").read_text(
            encoding="utf-8"
        )
    )

    staged_urdf, assets = _stage_urdf()
    robot_spec = mujoco.MjSpec.from_file(str(staged_urdf), assets=assets)
    robot_spec.worldbody.pos = np.asarray(ROBOT_POS, dtype=np.float64)
    robot_model = robot_spec.compile()
    _add_position_actuators(robot_spec, robot_model)
    spec = robot_spec

    # work table (static box, gs_gym TABLE_SIZE / TABLE_POS)
    table = spec.worldbody.add_body()
    table.name = "worktable"
    table.pos = np.asarray(TABLE_POS, dtype=np.float64)
    table_geom = table.add_geom()
    table_geom.name = "worktable_geom"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = np.asarray(TABLE_SIZE, dtype=np.float64) / 2.0
    table_geom.rgba = np.array([0.2, 0.2, 0.2, 1.0])
    table_geom.friction = np.array([1.0, 0.1, 0.1])

    # frozen objects: free boxes at the instance scene poses
    object_entries: dict[str, dict[str, Any]] = {}
    scene = inst["scene"]
    for name, entry in scene.items():
        info = (inst.get("objects") or {}).get(name) or {}
        size = info.get("size")
        pos = entry["pos"]
        quat = _euler_to_quat_wxyz(entry["euler"])
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = np.asarray(pos, dtype=np.float64)
        body.quat = quat
        geom = body.add_geom()
        geom.name = f"{name}__geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = np.asarray(size, dtype=np.float64) / 2.0
        geom.rgba = np.array([0.8, 0.5, 0.3, 1.0])
        geom.friction = np.array([1.0, 0.1, 0.1])
        if not info.get("fixed", False):
            joint = body.add_joint()
            joint.name = f"{name}__free"
            joint.type = mujoco.mjtJoint.mjJNT_FREE
        object_entries[name] = {
            "position": [float(v) for v in pos],
            "orientation_wxyz": [float(v) for v in quat],
            "size": [float(v) for v in size],
            "fixed": bool(info.get("fixed", False)),
        }

    xml = _absolutize_xml(spec.to_xml())
    model = mujoco.MjModel.from_xml_string(xml)

    # frozen state: robot reset joints (16 -> 18 dofs), then object free poses
    qpos = np.zeros(model.nq, dtype=np.float64)
    reset16 = np.asarray(inst["reset_joint_targets"], dtype=np.float64)
    layout = (
        list(reset16[0:7])
        + [reset16[7] / 2.0, reset16[7] / 2.0]
        + list(reset16[8:15])
        + [reset16[15] / 2.0, reset16[15] / 2.0]
    )
    qpos[: len(layout)] = layout
    adr = len(layout)
    for name, entry in object_entries.items():
        if entry["fixed"]:
            continue
        qpos[adr : adr + 3] = entry["position"]
        qpos[adr + 3 : adr + 7] = entry["orientation_wxyz"]
        adr += 7
    qvel = np.zeros(model.nv, dtype=np.float64)

    import hashlib

    state_sha = hashlib.sha1(np.ascontiguousarray(qpos).tobytes()).hexdigest()[:12]
    meta: dict[str, Any] = {
        "suite": "robowits",
        "task": task,
        "family": family,
        "text": FAMILIES[family]["text"],
        "robowits_task": inst["robowits_task"],
        "action_dim": 16,
        "control_hz": 30,
        "state_sha": state_sha,
        "instance_state_sha": inst.get("state_sha"),
        "objects": object_entries,
        "graded": {
            "rest": FAMILIES[family]["rest"],
            "release_also": FAMILIES[family]["release_also"],
        },
        "settle": {
            "settle_steps": SETTLE_STEPS,
            "check_steps": CHECK_STEPS,
            "check_stride": CHECK_STRIDE,
            "rest_m": REST_M,
        },
        "gripper_open": GRIPPER_OPEN,
        "table_z": TABLE_Z,
        "table_bounds": TABLE_BOUNDS,
        "ee_links": EE_LINKS,
        "robot_joints": ARM_JOINTS + GRIP_JOINTS,
        "upstream": {
            "robowits": "9cc30aedbbebea86b975c0d66ea9ac227ca38535",
            "assets": "XHRlyb2001/RoboWits_assets@b1b49abb",
        },
    }
    (WORLDS / f"{task}.xml").write_text(xml, encoding="utf-8")
    np.save(WORLDS / f"{task}.qpos.npy", qpos)
    np.save(WORLDS / f"{task}.qvel.npy", qvel)
    (WORLDS / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{task}: family={family} nq={model.nq} objects={len(object_entries)}")
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if not args.all and not args.task:
        raise SystemExit("give --task <name> or --all")

    if args.all:
        names = sorted(
            p.parent.parent.name.removesuffix("-privileged")
            for p in TASKS.glob("*-privileged/environment/instance.json")
        )
        names = [
            n
            for n in names
            if n.removesuffix("-i00").removeprefix("robowits-").replace("-", "_")
            in FAMILIES
        ]
    else:
        names = [args.task]
    failures = 0
    for name in names:
        try:
            compose_one(name)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"done: {len(names) - failures} composed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
