#!/usr/bin/env python3
"""Compose Genesis-ready DexToolBench world packages from the EFE instances.

DexToolBench's EFE tasks carry only ``environment/instance.json``
(``start_pose`` + ordered ``goals`` + ``pose_format x y z qx qy qz qw``);
the MuJoCo scene is rebuilt by SimToolReal's sim2sim recipe
(``deployment/mujoco/mujoco_sim.py::_init_scene``, mirrored at
``data/efe/upstream/dtb-src/simtoolreal``):

* menagerie ``kuka_iiwa_14/scene.xml``, Sharpa HA4 hand
  (``left_sharpa_ha4_v2_1_offset.xml``) attached at ``attachment_site``,
  robot base moved to (0, 0.8, 0);
* a 0.475 x 0.4 x 0.3 m table box centred at (0, 0, 0.38);
* the tool object (offline-CoACD-decomposed URDF) as a free body named
  ``object`` at the instance's start pose, friction [1.0, 0.005, 1e-4],
  ``condim=6`` on every geom, physics at 1 kHz like upstream.

This tool follows that recipe in the probe venv (MuJoCo MjSpec), pins the
grasp-box constants each task's ``instruction.md`` states verbatim
(``grasp box (A x B x C m, scaled by S)`` / corners within 1.5 cm), and
writes under ``assets/dextoolbench/worlds/``:

* ``<task>.xml`` — the composed world (asset paths rewritten to the local
  mirror roots);
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen state: iiwa
  ``IIWA_INIT_JOINT_POS``, 22 zero Sharpa joints, object free pose at
  ``start_pose``;
* ``<task>.meta.json`` — goals, grasp box, tolerance, joint/actuator
  inventory, the object body pose, and the upstream fingerprint.

One additive, physics-neutral extension over the upstream recipe: the
object's original visual mesh is added as a visual-only geom (contacts
off) — upstream renders the grey CoACD parts; grading uses the identical
collision geometry either way.

Usage::

    .abcvenv/Scripts/python.exe tools/dtb_worlds.py --all
    .abcvenv/Scripts/python.exe tools/dtb_worlds.py --task dextoolbench_brush_blue_brush_sweep_forward
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "efe" / "upstream" / "dtb-src"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "dextoolbench"
WORLDS = ROOT / "assets" / "dextoolbench" / "worlds"
ASSETS = ROOT / "assets" / "dextoolbench"

IIWA_SCENE = ASSETS / "menagerie" / "kuka_iiwa_14" / "scene.xml"
SHARPA_XML = (
    ASSETS
    / "simtoolreal"
    / "assets"
    / "urdf"
    / "left_sharpa_ha4"
    / "left_sharpa_ha4_v2_1_offset.xml"
)
OBJECTS_ROOT = ASSETS / "simtoolreal" / "assets" / "urdf" / "dextoolbench"

IIWA_INIT_JOINT_POS = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308])
SHARPA_N_JOINTS = 22
ROBOT_BASE_POS = (0.0, 0.8, 0.0)
TABLE_POS = (0.0, 0.0, 0.38)
TABLE_HALF = (0.475 / 2, 0.4 / 2, 0.3 / 2)
FRICTION = (1.0, 0.005, 0.0001)
TIMESTEP = 1.0 / 1000.0
OBJECT_RGBA = (0.5, 0.5, 0.5, 1.0)

_BOX_RE = re.compile(
    r"grasp box \(([\d.]+)\s*×\s*([\d.]+)\s*×\s*([\d.]+)\s*m,\s*scaled by\s*([\d.]+)",
    re.S,
)
_TOL_RE = re.compile(r"within\s*(?:\*\*)?([\d.]+)\s*cm", re.S)


def _absolutize_meshes(spec: Any, roots: list[Path]) -> None:
    """Rewrite every mesh asset path to an absolute mirror path.

    ``MjSpec.to_xml`` keeps mesh refs as written in each source file
    (relative to that file's dir), which collides once several sources are
    merged — resolve them against their source roots before attaching.
    """
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        candidate = Path(mesh.file)
        if candidate.is_file():
            continue
        rel = candidate.as_posix().lstrip("./")
        for root in roots:
            for base in (root, root / "assets", root / "meshes"):
                resolved = base / rel
                if resolved.is_file():
                    mesh.file = str(resolved)
                    break
            if Path(mesh.file).is_file():
                break


def _load_instance(task_dir: Path) -> dict[str, Any]:
    for rel in ("environment/instance.json", "environment/sim/instance.json"):
        f = task_dir / rel
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    raise SystemExit(f"no instance.json in {task_dir}")


def _parse_grasp_box(instruction: Path) -> tuple[list[float], float, float]:
    """Grasp-box dims (metres), the instruction's scale factor, corner tolerance."""
    text = instruction.read_text(encoding="utf-8")
    box = _BOX_RE.search(text)
    tol = _TOL_RE.search(text)
    if box is None or tol is None:
        raise SystemExit(f"grasp box/tolerance not found in {instruction}")
    dims = [float(v) for v in box.groups()[:3]]
    scale = float(box.group(4))
    return dims, scale, float(tol.group(1)) / 100.0


def _decomposed_geoms(object_name: str, category: str) -> tuple[Path, list[Path]]:
    """Original visual mesh and the CoACD collision meshes of one tool."""
    obj_dir = OBJECTS_ROOT / category / object_name
    decomposed = obj_dir / f"{object_name}_decomposed.urdf"
    if not decomposed.is_file():
        raise SystemExit(f"decomposed URDF missing: {decomposed}")
    root = ET.fromstring(decomposed.read_text(encoding="utf-8"))
    visual = None
    collisions: list[Path] = []
    for coll in root.iter("collision"):
        mesh = coll.find("geometry/mesh")
        if mesh is not None and mesh.get("filename"):
            collisions.append(obj_dir / mesh.get("filename", ""))
    for vis in root.iter("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is not None and mesh.get("filename"):
            visual = obj_dir / mesh.get("filename", "")
            break
    if not collisions:
        raise SystemExit(f"no collision meshes in {decomposed}")
    return visual or collisions[0], collisions


def _pose_to_pos_quat(pose: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """instance.json pose (x y z qx qy qz qw) -> (pos, wxyz quat)."""
    if len(pose) != 7:
        raise SystemExit(f"pose must have 7 numbers, got {pose!r}")
    pos = np.asarray(pose[:3], dtype=np.float64)
    quat = np.asarray([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)
    return pos, quat


def extract_one(task: str, check_assets: bool) -> dict[str, Any]:
    """Compose one DexToolBench world package; return its meta."""
    import mujoco

    task_dir = TASKS / f"{task}-privileged"
    inst = _load_instance(task_dir)
    dims, box_scale, tolerance = _parse_grasp_box(task_dir / "instruction.md")
    category, object_name = inst["category"], inst["object"]
    visual_mesh, collision_meshes = _decomposed_geoms(object_name, category)
    start_pos, start_quat = _pose_to_pos_quat(inst["start_pose"])
    goals = [[float(v) for v in g] for g in inst["goals"]]

    spec = mujoco.MjSpec.from_file(str(IIWA_SCENE))
    spec.option.timestep = TIMESTEP
    sharpa_spec = mujoco.MjSpec.from_file(str(SHARPA_XML))
    attachment_site = next(s for s in spec.sites if s.name == "attachment_site")
    attachment_site.attach_body(sharpa_spec.worldbody, "palm", "")

    _absolutize_meshes(spec, [IIWA_SCENE.parent])
    _absolutize_meshes(sharpa_spec, [SHARPA_XML.parent])
    robot_base = next(b for b in spec.bodies if b.name == "base")
    robot_base.pos = np.asarray(ROBOT_BASE_POS, dtype=np.float64)

    table_body = spec.worldbody.add_body()
    table_body.name = "table"
    table_body.pos = np.asarray(TABLE_POS, dtype=np.float64)
    for suffix in ("visual", "collision"):
        geom = table_body.add_geom()
        geom.name = f"table_geom_{suffix}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = np.asarray(TABLE_HALF, dtype=np.float64)
        geom.rgba = np.array([1.0, 1.0, 1.0, 1.0])
        geom.friction = np.asarray(FRICTION, dtype=np.float64)
        if suffix == "visual":
            geom.contype = 0
            geom.conaffinity = 0

    object_body = spec.worldbody.add_body()
    object_body.name = "object"
    object_body.pos = start_pos
    object_body.quat = start_quat
    free_joint = object_body.add_joint()
    free_joint.name = "object_free_joint"
    free_joint.type = mujoco.mjtJoint.mjJNT_FREE

    visual_mesh_asset = spec.add_mesh()
    visual_mesh_asset.name = "object_visual_mesh"
    visual_mesh_asset.file = str(visual_mesh)
    visual_mesh_asset.scale = np.array([1.0, 1.0, 1.0])
    visual = object_body.add_geom()
    visual.name = "object_visual"
    visual.type = mujoco.mjtGeom.mjGEOM_MESH
    visual.meshname = visual_mesh_asset.name
    visual.rgba = np.asarray(OBJECT_RGBA, dtype=np.float64)
    visual.contype = 0
    visual.conaffinity = 0
    for index, mesh_path in enumerate(collision_meshes):
        mesh = spec.add_mesh()
        mesh.name = f"object_mesh_{index}"
        mesh.file = str(mesh_path)
        mesh.scale = np.array([1.0, 1.0, 1.0])
        geom = object_body.add_geom()
        geom.name = f"object_geom_{index}"
        geom.rgba = np.asarray(OBJECT_RGBA, dtype=np.float64)
        geom.friction = np.asarray(FRICTION, dtype=np.float64)
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name

    for geom in spec.geoms:
        geom.condim = 6

    model = spec.compile()
    xml = spec.to_xml()

    nq = int(model.nq)
    nv = int(model.nv)
    arm_n = len(IIWA_INIT_JOINT_POS)
    object_adr = nq - 7
    qpos = np.zeros(nq, dtype=np.float64)
    qpos[:arm_n] = IIWA_INIT_JOINT_POS
    qpos[object_adr : object_adr + 3] = start_pos
    qpos[object_adr + 3 : object_adr + 7] = start_quat

    joints = []
    cursor = 0
    for i in range(model.njnt):
        name = model.jnt(i).name
        jtype = int(model.jnt_type[i])
        width = 7 if jtype == 0 else (4 if jtype == 1 else 1)
        joints.append(
            {
                "name": name,
                "type": {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(
                    jtype, str(jtype)
                ),
                "qposadr": int(model.jnt_qposadr[i]),
                "qveladr": int(model.jnt_dofadr[i]),
                "body": model.body(int(model.jnt_bodyid[i])).name,
            }
        )
        cursor += width
    actuators = [
        {
            "name": model.actuator(i).name,
            "joint": model.joint(int(model.actuator_trnid[i][0])).name,
            "ctrlrange": [float(v) for v in model.actuator_ctrlrange[i]],
        }
        for i in range(model.nu)
    ]

    out_dir = WORLDS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}.xml").write_text(xml, encoding="utf-8")
    np.save(out_dir / f"{task}.qpos.npy", qpos)
    np.save(out_dir / f"{task}.qvel.npy", np.zeros(nv, dtype=np.float64))

    meta = {
        "suite": "dextoolbench",
        "task": task,
        "category": category,
        "object": object_name,
        "task_name": inst["task"],
        "text": inst.get("text", ""),
        "goals": goals,
        "pose_format": inst.get("pose_format", "x y z qx qy qz qw"),
        "grasp_box_dims_m": dims,
        "grasp_box_scale": box_scale,
        "corner_tolerance_m": tolerance,
        "nq": nq,
        "nv": nv,
        "nu": int(model.nu),
        "object_body": "object",
        "object_start_pos": [float(v) for v in start_pos],
        "object_start_quat": [float(v) for v in start_quat],
        "joints": joints,
        "actuators": actuators,
        "upstream": {
            "simtoolreal": "313d5ae",
            "menagerie": "1b86ece576591213e2b666ebf59508454200ca97",
            "recipe": "deployment/mujoco/mujoco_sim.py::_init_scene",
        },
    }
    (out_dir / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    if check_assets:
        missing = sorted(
            {m for m in re.findall(r'file="([^"]+)"', xml) if not Path(m).is_file()}
        )  # paths are absolute post-absolutize
        if missing:
            print(f"WARNING {task}: {len(missing)} missing assets", file=sys.stderr)
            for m in missing[:6]:
                print(f"   {m}", file=sys.stderr)
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-check", action="store_true")
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
            meta = extract_one(name, check_assets=not args.no_check)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {name}: {exc}", file=sys.stderr)
            continue
        print(
            f"{name}: goals={len(meta['goals'])} box={meta['grasp_box_dims_m']}"
            f"x{meta['grasp_box_scale']} nq={meta['nq']} nu={meta['nu']}"
        )
    print(f"done: {len(names) - failures} composed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
