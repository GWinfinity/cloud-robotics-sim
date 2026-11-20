#!/usr/bin/env python3
"""Compose Genesis-ready MuJoCo Playground world packages from the EFE instances.

The 13 Playground tasks (9 quadruped locomotion + 4 manipulation) are rebuilt
exactly as the EFE graders' sessions build them
(``data/efe/upstream/pg-src/efe-benchmark/{rcb_quadruped,rcb_aloha,rcb_panda}.py``,
mirrored from ``embodied_first_exam/benchmarks/mujoco-playground``):

* quadrupeds — Playground's ``locomotion/<robot>/xmls`` scene with the
  Menagerie robot, loaded with a bare-file-name asset dict (Playground's
  ``get_assets()`` way); Barkour additionally gets the foot-floor contact
  sensor per foot its joystick env adds; drive gains as Playground's
  environments set them (``Kp`` actuator gain / ``-Kp`` bias, ``Kd`` joint
  damping from index 6); the frozen start is the instance's ``qpos0`` /
  ``qvel0`` with ``ctrl = qpos0[7:]`` and, for getup, Playground's 0.5 s
  settle with the joints held;
* manipulation — ``manipulation/{aloha,franka_emika_panda}/xmls`` with the
  Menagerie robot, the env's sim timestep, the ``home``/``upright`` keyframe,
  the instance's randomization (peg/socket or box offsets, arm perturbation),
  PandaOpenCabinet's ``hand_capsule`` conaffinity 3, and the translucent
  mocap target demoted to a static visual marker (its pose is frozen at the
  instance target; upstream reads the target from the instance, never from
  the mocap).

Asset file references (meshes / textures / heightfields) are rewritten to
absolute mirror paths so the composed XML is self-contained. Writes under
``assets/playground/worlds/``:

* ``<task>.xml`` — the composed world;
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen t = 0 state;
* ``<task>.meta.json`` — kind-specific check constants (segments, kicks,
  gait, termination boxes, site/geom names, peg axial limits, targets).

Usage::

    .abcvenv/Scripts/python.exe tools/pg_worlds.py --all
    .abcvenv/Scripts/python.exe tools/pg_worlds.py --task mujoco_playground_go1_getup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "playground"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "mujoco-playground"
WORLDS = ASSETS / "worlds"

#: rcb_quadruped.py verbatim -------------------------------------------------
CONTROL_DT = 0.02
SETTLE_STEPS = 25
LIN_TOL = 0.2
YAW_TOL = 0.3
HOLD_STEPS = 25
TILT_MAX_DEG = 10.0
GAIT_TOL = 0.05
POSE_HOLD_STEPS = 100
FRONT_LEG_GEOMS = [
    "fl_calf1",
    "fl_calf2",
    "fr_calf1",
    "fr_calf2",
    "fl_thigh1",
    "fl_thigh2",
    "fl_thigh3",
    "fr_thigh1",
    "fr_thigh2",
    "fr_thigh3",
    "fl_hip",
    "fr_hip",
]
HIND_LEG_GEOMS = [
    g.replace("fl_", "rl_").replace("fr_", "rr_") for g in FRONT_LEG_GEOMS
]

ROBOTS = {
    "go1": {
        "assets": [
            ("playground/mujoco_playground/_src/locomotion/go1/xmls", "*.xml"),
            ("playground/mujoco_playground/_src/locomotion/go1/xmls/assets", "*"),
            ("menagerie/unitree_go1", "*.xml"),
            ("menagerie/unitree_go1/assets", "*"),
        ],
        "imu": "imu",
        "feet": ["FR", "FL", "RR", "RL"],
        "root": "trunk",
    },
    "spot": {
        "assets": [
            ("playground/mujoco_playground/_src/locomotion/spot/xmls", "*.xml"),
            ("menagerie/boston_dynamics_spot", "*.xml"),
            ("menagerie/boston_dynamics_spot/assets", "*"),
        ],
        "imu": "imu",
        "feet": ["FL", "FR", "HL", "HR"],
        "root": "body",
    },
    "barkour": {
        "assets": [
            ("menagerie/google_barkour_vb", "*.xml"),
            ("menagerie/google_barkour_vb/assets", "*"),
        ],
        "imu": "imu_frame",
        "feet": [
            "foot_front_left",
            "foot_hind_left",
            "foot_front_right",
            "foot_hind_right",
        ],
        "root": "torso",
    },
}

QUADRUPED_ENVS = {
    "Go1JoystickFlatTerrain": {
        "robot": "go1",
        "kind": "joystick",
        "horizon": 1000,
        "kp": 35.0,
        "kd": 0.5,
        "xml": "playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml",
    },
    "Go1JoystickRoughTerrain": {
        "robot": "go1",
        "kind": "joystick",
        "horizon": 1000,
        "kp": 35.0,
        "kd": 0.5,
        "xml": "playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_rough_terrain.xml",
    },
    "Go1Getup": {
        "robot": "go1",
        "kind": "getup",
        "horizon": 300,
        "kp": 35.0,
        "kd": 0.5,
        "settle_s": 0.5,
        "z_des": 0.275,
        "min_imu_height": 0.22,
        "xml": "playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_fullcollisions_flat_terrain.xml",
    },
    "Go1Handstand": {
        "robot": "go1",
        "kind": "stand_on_two",
        "horizon": 500,
        "kp": 35.0,
        "kd": 0.5,
        "forward": [0.0, 0.0, -1.0],
        "air_feet": ["RR", "RL"],
        "fwd_tol_deg": 30.0,
        "min_imu_height": 0.40,
        "xml": "playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_flat_terrain.xml",
    },
    "Go1Footstand": {
        "robot": "go1",
        "kind": "stand_on_two",
        "horizon": 500,
        "kp": 35.0,
        "kd": 0.5,
        "forward": [0.0, 0.0, 1.0],
        "air_feet": ["FR", "FL"],
        "fwd_tol_deg": 30.0,
        "min_imu_height": 0.40,
        "hind_leg_contact": True,
        "xml": "playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_flat_terrain.xml",
    },
    "SpotGetup": {
        "robot": "spot",
        "kind": "getup",
        "horizon": 300,
        "kp": 400.0,
        "kd": 20.0,
        "settle_s": 0.5,
        "z_des": 0.46,
        "min_imu_height": 0.40,
        "xml": "playground/mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_flat_terrain.xml",
    },
    "SpotFlatTerrainJoystick": {
        "robot": "spot",
        "kind": "joystick",
        "horizon": 1000,
        "kp": 300.0,
        "kd": 1.0,
        "up_z_min": 0.85,
        "xml": "playground/mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_feetonly_flat_terrain.xml",
    },
    "SpotJoystickGaitTracking": {
        "robot": "spot",
        "kind": "joystick",
        "horizon": 1000,
        "kp": 400.0,
        "kd": 10.0,
        "up_z_min": 0.85,
        "gait": True,
        "xml": "playground/mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_feetonly_flat_terrain.xml",
    },
    "BarkourJoystick": {
        "robot": "barkour",
        "kind": "joystick",
        "horizon": 1000,
        "kp": 35.0,
        "kd": 0.5239,
        "up_z_min": 0.0,
        "control_dt": 0.01,
        "joint_box": [[-0.7, -1.0, 0.05], [0.52, 2.1, 2.1]],
        "torso_min_z": 0.18,
        "kicks": True,
        "xml": "menagerie/google_barkour_vb/scene_mjx.xml",
    },
}

#: rcb_aloha.py / rcb_panda.py verbatim ---------------------------------------
ALOHA_ENVS = {
    "AlohaSinglePegInsertion": {
        "xml": "mjx_single_peg_insertion.xml",
        "sim_dt": 0.0025,
        "task": "mujoco_playground_aloha_single_peg_insertion",
    },
    "AlohaHandOver": {
        "xml": "mjx_hand_over.xml",
        "sim_dt": 0.005,
        "task": "mujoco_playground_aloha_hand_over",
    },
}
ALOHA_ACTUATORS = [
    f"{s}/{j}"
    for s in ("left", "right")
    for j in (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    )
]
PANDA_ENVS = {
    "PandaPickCubeOrientation": {
        "xml": "mjx_single_cube.xml",
        "key": "home",
        "task": "mujoco_playground_panda_pick_cube_orientation",
        "obj": "box",
    },
    "PandaOpenCabinet": {
        "xml": "mjx_cabinet.xml",
        "key": "upright",
        "task": "mujoco_playground_panda_open_cabinet",
        "obj": "handle",
    },
}
PANDA_ACTUATORS = [f"actuator{i}" for i in range(1, 9)]
HANDOVER_TOL_M = 0.03
LINE_TOL_M = 0.005
DEPTH_T_MAX = 0.75
REAR_MARGIN_M = 0.002
PICK_POS_TOL = 0.02
PICK_ROT_TOL_DEG = 15.0
CABINET_TOL = 0.015


def _assets_dict(robot: str) -> dict[str, bytes]:
    """Bare-file-name asset dict the way Playground's get_assets() builds it."""
    assets: dict[str, bytes] = {}
    for d, pat in ROBOTS[robot]["assets"]:
        for f in sorted((ASSETS / d).glob(pat)):
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets


def _absolutize_xml(xml: str, xml_dir: Path) -> str:
    """Rewrite every asset file ref (mesh/texture face/hfield) to absolute.

    MuJoCo MjSpec does not expose cube-texture face files
    (``fileright=`` ...) as Python attributes, so the rewrite runs on the
    exported XML text: any ``file*="<path>"`` attribute whose path resolves
    inside the mirror becomes an absolute path.
    """
    import re

    roots = [
        xml_dir,
        xml_dir / "assets",
        ASSETS,
        ASSETS / "menagerie" / "aloha",
        ASSETS / "menagerie" / "franka_emika_panda",
        ASSETS / "menagerie" / "unitree_go1",
        ASSETS / "menagerie" / "boston_dynamics_spot",
        ASSETS / "menagerie" / "google_barkour_vb",
    ]

    def _resolve(match: re.Match[str]) -> str:
        attr, ref = match.group(1), match.group(2)
        if not ref or ref.startswith(("http://", "https://", "data:")):
            return match.group(0)
        if "\\" in ref:
            posix = ref.replace("\\", "/")
            if Path(posix).is_file():
                return f'{attr}="{posix}"'
        if Path(ref).is_file() and "/" in ref:
            return match.group(0)
        rel = ref.lstrip("./")
        for base in roots:
            cand = base / rel
            if cand.is_file():
                return f'{attr}="{cand.as_posix()}"'
        hits = [f for f in ASSETS.rglob(Path(rel).name) if f.is_file()]
        if hits:
            return f'{attr}="{hits[0].as_posix()}"'
        raise SystemExit(f"asset not found in mirror: {rel}")

    return re.sub(
        r'(\sfile(?:right|left|up|down|front|back)?)="([^"]+)(")', _resolve, xml
    )


def _spec_to_xml(spec: Any, assets: dict[str, bytes]) -> str:
    """Export a spec loaded from memory; bare-name asset files materialized.

    ``MjSpec.to_xml`` re-opens every mesh/texture file at its recorded
    (meshdir-relative) path, so the bare-file-name asset dict Playground
    loads with is materialized in a temp dir for the export.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, content in assets.items():
            for base in (root, root / "assets"):
                base.mkdir(parents=True, exist_ok=True)
                (base / name).write_bytes(content)
        cwd = Path.cwd()
        os.chdir(root)
        try:
            return spec.to_xml()
        finally:
            os.chdir(cwd)


def _apply_gains(spec: Any, kp: float, kd: float) -> None:
    """Playground's drive gains: Kp as the actuator gain/-bias, Kd as damping.

    ``dof_damping[6:] = kd`` in the session code covers every DoF after the
    floating base; on the spec that is every non-free joint (each maps to one
    hinge/slide DoF).
    """
    import mujoco

    for act in spec.actuators:
        gain = np.asarray(act.gainprm, dtype=np.float64).copy()
        gain[0] = kp
        act.gainprm = gain
        bias = np.asarray(act.biasprm, dtype=np.float64).copy()
        bias[1] = -kp
        act.biasprm = bias
    for joint in spec.joints:
        if int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        joint.damping = float(kd)


def _state_sha(qpos: np.ndarray, qvel: np.ndarray) -> str:
    return hashlib.sha1(np.concatenate([qpos, qvel]).tobytes()).hexdigest()[:12]


def _load_instance(task: str) -> dict[str, Any]:
    for cand in (TASKS / f"{task}-privileged", TASKS / task):
        f = cand / "environment" / "instance.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    raise SystemExit(f"no instance.json mirrored for {task}")


# --------------------------------------------------------------------------- quadrupeds


def compose_quadruped(task: str, inst: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one quadruped world exactly as rcb_quadruped.Session does."""
    import mujoco

    env = inst["env"]
    spec_consts = QUADRUPED_ENVS[env]
    robot = ROBOTS[spec_consts["robot"]]
    assets = _assets_dict(spec_consts["robot"])

    spec = mujoco.MjSpec.from_file(str(ASSETS / spec_consts["xml"]), assets=assets)
    if spec_consts["robot"] == "barkour":
        for g in robot["feet"]:
            spec.add_sensor(
                name=f"{g}_floor_found",
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname=g,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname="floor",
                intprm=[1, 1, 1],
                datatype=mujoco.mjtDataType.mjDATATYPE_REAL,
                needstage=mujoco.mjtStage.mjSTAGE_ACC,
                dim=1,
            )
    _apply_gains(spec, float(spec_consts["kp"]), float(spec_consts["kd"]))
    xml_dir = (ASSETS / spec_consts["xml"]).parent
    xml = _absolutize_xml(spec.to_xml(), xml_dir)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = np.asarray(inst["qpos0"], dtype=np.float64)
    data.qvel[:] = np.asarray(inst["qvel0"], dtype=np.float64)
    data.ctrl[:] = np.asarray(inst["qpos0"], dtype=np.float64)[7:]
    mujoco.mj_forward(model, data)
    if spec_consts.get("settle_s"):
        mujoco.mj_step(
            model, data, nstep=int(float(spec_consts["settle_s"]) / model.opt.timestep)
        )
        data.time = 0.0
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    joint_names = [
        model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)
    ]
    ctrlrange = [
        [float(v) for v in model.actuator_ctrlrange[i]] for i in range(model.nu)
    ]

    def _site_record(site_name: str) -> dict[str, Any]:
        sid = model.site(site_name).id
        body_id = int(model.site_bodyid[sid])
        return {
            "body": model.body(body_id).name,
            "local": [float(v) for v in model.site_pos[sid]],
            "local_quat": [float(v) for v in model.site_quat[sid]],
        }

    sites = {robot["imu"]: _site_record(robot["imu"])}
    for f in robot["feet"]:
        sites[f] = _site_record(f)
    meta: dict[str, Any] = {
        "suite": "mujoco-playground",
        "task": task,
        "env": env,
        "kind_group": "quadruped",
        "robot": spec_consts["robot"],
        "task_kind": spec_consts["kind"],
        "text": inst.get("text", ""),
        "seed": inst.get("seed"),
        "horizon": int(spec_consts["horizon"]),
        "control_dt": float(spec_consts.get("control_dt", CONTROL_DT)),
        "timestep": float(model.opt.timestep),
        "imu_site": robot["imu"],
        "root_body": robot["root"],
        "floor_geom": "floor",
        "feet": list(robot["feet"]),
        "sites": sites,
        "joint_names": joint_names,
        "ctrlrange": ctrlrange,
        "home_qpos": [float(v) for v in model.keyframe("home").qpos[7:]],
        "constants": {
            "settle_steps": SETTLE_STEPS,
            "lin_tol": LIN_TOL,
            "yaw_tol": YAW_TOL,
            "hold_steps": HOLD_STEPS,
            "tilt_max_deg": TILT_MAX_DEG,
            "gait_tol": GAIT_TOL,
            "pose_hold_steps": POSE_HOLD_STEPS,
            "front_leg_geoms": FRONT_LEG_GEOMS,
            "hind_leg_geoms": HIND_LEG_GEOMS,
            "up_z_min": float(spec_consts.get("up_z_min", 0.0)),
            "joint_box": spec_consts.get("joint_box"),
            "torso_min_z": spec_consts.get("torso_min_z"),
            "min_imu_height": float(spec_consts.get("min_imu_height", 0.0)) or None,
            "forward": spec_consts.get("forward"),
            "fwd_tol_deg": spec_consts.get("fwd_tol_deg"),
            "air_feet": spec_consts.get("air_feet"),
            "hind_leg_contact": bool(spec_consts.get("hind_leg_contact", False)),
            "z_des": spec_consts.get("z_des"),
        },
        "commands": inst.get("commands") if spec_consts["kind"] == "joystick" else None,
        "kicks": inst.get("kicks") if spec_consts.get("kicks") else [],
        "gait": inst.get("gait") if spec_consts.get("gait") else None,
        "settle_s": spec_consts.get("settle_s", 0.0),
        "state_sha": _state_sha(qpos, qvel),
        "instance_state_sha": inst.get("state_sha"),
        "upstream": {
            "playground": "4057c14",
            "menagerie": "1b86ece",
            "recipe": "efe-benchmark/rcb_quadruped.py::Session",
        },
    }
    return {"xml": xml, "qpos": qpos, "qvel": qvel, "meta": meta}


# --------------------------------------------------------------------------- manipulation


def _demote_mocap(
    spec: Any, target_pos: list[float], target_quat: list[float] | None
) -> None:
    """Translucent mocap target -> static visual marker at the instance pose.

    The graders read the target from the instance (rcb_panda / rcb_aloha keep
    ``target_pos`` from instance.json); the mocap body is render-only, so the
    world pack freezes it as a static body — physics-neutral.
    """
    for body in spec.bodies:
        if not getattr(body, "mocap", False):
            continue
        body.mocap = False
        body.pos = np.asarray(target_pos, dtype=np.float64)
        if target_quat is not None:
            body.quat = np.asarray(target_quat, dtype=np.float64)


def _geom_name_sets(model: Any) -> tuple[list[str], list[str]]:
    """rcb_aloha handover classification as name lists (right fingers / left)."""
    right, left = [], []
    for g in range(model.ngeom):
        bname = model.body(int(model.geom_bodyid[g])).name
        gname = model.geom(g).name
        if not gname:
            continue
        if bname.startswith("right/") and "finger" in bname:
            right.append(gname)
        if bname.startswith("left/"):
            left.append(gname)
    return right, left


def compose_aloha(task: str, inst: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one ALOHA world exactly as rcb_aloha.Session does."""
    import mujoco

    env = inst["env"]
    spec_consts = ALOHA_ENVS[env]
    xml_dir = (
        ASSETS
        / "playground"
        / "mujoco_playground"
        / "_src"
        / "manipulation"
        / "aloha"
        / "xmls"
    )
    assets: dict[str, bytes] = {}
    for d in (
        ASSETS / "menagerie" / "aloha",
        ASSETS / "menagerie" / "aloha" / "assets",
        xml_dir,
        xml_dir / "assets",
    ):
        for f in d.glob("*"):
            if f.is_file():
                assets[f.name] = f.read_bytes()
    spec = mujoco.MjSpec.from_string(
        (xml_dir / spec_consts["xml"]).read_text(encoding="utf-8"), assets
    )
    spec.option.timestep = float(spec_consts["sim_dt"])
    if env == "AlohaHandOver":
        _demote_mocap(spec, inst["target_pos"], None)
    xml = _absolutize_xml(_spec_to_xml(spec, assets), xml_dir)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    if env == "AlohaSinglePegInsertion":
        peg = model.body("peg").id
        sock = model.body("socket").id
        pq = int(model.jnt_qposadr[int(model.body_jntadr[peg])])
        sq = int(model.jnt_qposadr[int(model.body_jntadr[sock])])
        data.qpos[pq : pq + 2] += np.asarray(inst["peg_xy_offset"], dtype=np.float64)
        data.qpos[sq : sq + 2] += np.asarray(inst["socket_xy_offset"], dtype=np.float64)
    else:
        box = model.body("box").id
        bq = int(model.jnt_qposadr[int(model.body_jntadr[box])])
        data.qpos[bq : bq + 2] += np.asarray(inst["box_xy_offset"], dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    meta: dict[str, Any] = {
        "suite": "mujoco-playground",
        "task": task,
        "env": env,
        "kind_group": "manipulation",
        "task_kind": env,
        "text": inst.get("text", ""),
        "seed": inst.get("seed"),
        "timestep": float(spec_consts["sim_dt"]),
        "control_dt": CONTROL_DT,
        "substeps": int(round(CONTROL_DT / float(spec_consts["sim_dt"]))),
        "actuators": ALOHA_ACTUATORS,
        "max_steps": 3000,
        "state_sha": _state_sha(qpos, qvel),
        "instance_state_sha": inst.get("state_sha"),
        "upstream": {
            "playground": "4057c14",
            "menagerie": "1b86ece",
            "recipe": "efe-benchmark/rcb_aloha.py::Session",
        },
    }
    if env == "AlohaSinglePegInsertion":
        s_ent = model.site("socket_entrance").id
        s_rear = model.site("socket_rear").id
        ax = model.site_pos[s_ent] - model.site_pos[s_rear]
        ln = float(np.linalg.norm(ax))
        u = ax / ln
        w = model.geom("wall").id
        rw = np.zeros(9)
        mujoco.mju_quat2Mat(rw, model.geom_quat[w])
        rw = rw.reshape(3, 3)
        face = float(
            np.dot(model.geom_pos[w] - model.site_pos[s_rear], u)
            + np.sum(np.abs(rw.T @ u) * model.geom_size[w])
        )

        def _site_record(site_name: str) -> dict[str, Any]:
            sid = model.site(site_name).id
            return {
                "body": model.body(int(model.site_bodyid[sid])).name,
                "local": [float(v) for v in model.site_pos[sid]],
                "local_quat": [float(v) for v in model.site_quat[sid]],
            }

        meta["sites"] = {
            n: _site_record(n) for n in ("socket_entrance", "socket_rear", "peg_end2")
        }
        meta["check"] = {
            "sites": {
                "socket_rear": "socket_rear",
                "socket_entrance": "socket_entrance",
                "peg_tip": "peg_end2",
            },
            "t_wall": face / ln,
            "t_min": (face - REAR_MARGIN_M) / ln,
            "depth_t_max": DEPTH_T_MAX,
            "line_tol_m": LINE_TOL_M,
        }
    else:
        right, left = _geom_name_sets(model)
        meta["check"] = {
            "box_body": "box",
            "box_geom": "box",
            "target_pos": [float(v) for v in inst["target_pos"]],
            "tol_m": HANDOVER_TOL_M,
            "min_z": 0.05,
            "right_finger_geoms": right,
            "left_geoms": left,
        }
    return {"xml": xml, "qpos": qpos, "qvel": qvel, "meta": meta}


def compose_panda(task: str, inst: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one Panda world exactly as rcb_panda.Session does."""
    import mujoco

    env = inst["env"]
    spec_consts = PANDA_ENVS[env]
    xml_dir = (
        ASSETS
        / "playground"
        / "mujoco_playground"
        / "_src"
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
    )
    assets: dict[str, bytes] = {}
    for d in (
        ASSETS / "menagerie" / "franka_emika_panda",
        ASSETS / "menagerie" / "franka_emika_panda" / "assets",
        xml_dir,
    ):
        for f in d.glob("*"):
            if f.is_file():
                assets[f.name] = f.read_bytes()
    spec = mujoco.MjSpec.from_string(
        (xml_dir / spec_consts["xml"]).read_text(encoding="utf-8"), assets
    )
    spec.option.timestep = 0.005
    if env == "PandaOpenCabinet":
        spec.geom("hand_capsule").conaffinity = 3
    _demote_mocap(spec, inst["target_pos"], inst.get("target_quat_wxyz"))
    xml = _absolutize_xml(_spec_to_xml(spec, assets), xml_dir)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(spec_consts["key"]).id)
    if env == "PandaPickCubeOrientation":
        qa = int(model.jnt_qposadr[int(model.body_jntadr[model.body("box").id])])
        data.qpos[qa : qa + 3] = np.asarray(inst["box_pos"], dtype=np.float64)
    else:
        pert = np.asarray(inst["arm_perturbation"], dtype=np.float64)
        data.qpos[:7] += pert
        data.ctrl[:7] += pert
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    meta: dict[str, Any] = {
        "suite": "mujoco-playground",
        "task": task,
        "env": env,
        "kind_group": "manipulation",
        "task_kind": env,
        "text": inst.get("text", ""),
        "seed": inst.get("seed"),
        "timestep": 0.005,
        "control_dt": CONTROL_DT,
        "substeps": 4,
        "actuators": PANDA_ACTUATORS,
        "max_steps": 3000,
        "object_body": spec_consts["obj"],
        "target_pos": [float(v) for v in inst["target_pos"]],
        "target_quat_wxyz": [
            float(v) for v in inst.get("target_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        ],
        "state_sha": _state_sha(qpos, qvel),
        "instance_state_sha": inst.get("state_sha"),
        "upstream": {
            "playground": "4057c14",
            "menagerie": "1b86ece",
            "recipe": "efe-benchmark/rcb_panda.py::Session",
        },
    }
    if env == "PandaPickCubeOrientation":
        meta["check"] = {"pos_tol_m": PICK_POS_TOL, "rot_tol_deg": PICK_ROT_TOL_DEG}
    else:
        meta["check"] = {"tol_m": CABINET_TOL}
    return {"xml": xml, "qpos": qpos, "qvel": qvel, "meta": meta}


# --------------------------------------------------------------------------- driver


def compose_one(task: str) -> dict[str, Any]:
    """Compose one Playground world package; return its meta."""
    inst = _load_instance(task)
    env = inst["env"]
    if env in QUADRUPED_ENVS:
        pack = compose_quadruped(task, inst)
    elif env in ALOHA_ENVS:
        pack = compose_aloha(task, inst)
    elif env in PANDA_ENVS:
        pack = compose_panda(task, inst)
    else:
        raise SystemExit(f"{task}: unknown env {env!r}")

    WORLDS.mkdir(parents=True, exist_ok=True)
    (WORLDS / f"{task}.xml").write_text(pack["xml"], encoding="utf-8")
    np.save(WORLDS / f"{task}.qpos.npy", pack["qpos"])
    np.save(WORLDS / f"{task}.qvel.npy", pack["qvel"])
    meta = pack["meta"]
    (WORLDS / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    sha_note = ""
    if (
        meta.get("instance_state_sha")
        and meta["state_sha"] != meta["instance_state_sha"]
    ):
        sha_note = (
            f" (probe sha {meta['state_sha']} != instance {meta['instance_state_sha']})"
        )
    print(f"{task}: env={env} nq={len(pack['qpos'])}{sha_note}")
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
    else:
        names = [args.task]
    failures = 0
    for name in names:
        try:
            compose_one(name)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {name}: {exc}", file=sys.stderr)
            raise
    print(f"done: {len(names) - failures} composed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
