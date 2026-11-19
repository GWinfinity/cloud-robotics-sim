#!/usr/bin/env python3
"""Extract Genesis-ready DexJoCo world packages from the EFE instances.

DexJoCo's EFE tasks freeze one demonstration's reset state: the env's
``arena_arm_hand_*.xml`` scene, every numeric ``MjModel`` array the state
restorers edited (``model__*`` members of ``instance.npz``), the full
``mjSTATE_INTEGRATION`` vector (``qpos | qvel | act``), mocap targets and
the env's Python-side attrs. This tool replays those overlays in the probe
venv (mujoco == 3.4.0, the version the suite's hashes were computed with):

* compile the family's env XML from the mirrored package;
* apply every ``model__*`` array onto the compiled model (in place, so the
  restorers' edits — table height, fixed bodies, actuator params — become
  the model's values);
* ``mj_saveLastXML`` the resulting model to one world XML per task, then
  absolutise every ``file="..."`` against the ``assets/dexjoco/xmls``
  mirror;
* split ``physics`` into ``<task>.qpos.npy`` / ``<task>.qvel.npy`` (act is
  empty for this suite) and stash mocap targets + attrs + the npz sha in
  ``<task>.meta.json``.

The npz records no scene XML name; the family -> XML map is DexJoCo's own
``_XML_PATH`` constants (mirrored under ``data/efe/upstream/djx-src``).

Usage::

    .djxvenv/Scripts/python.exe tools/djx_worlds.py --all
    .djxvenv/Scripts/python.exe tools/djx_worlds.py --task dexjoco-water-plant-i00
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "efe" / "upstream" / "djx-src" / "dexjoco"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "dexjoco"
WORLDS = ROOT / "assets" / "dexjoco" / "worlds"
XMLS = ROOT / "assets" / "dexjoco" / "xmls"

#: DexJoCo's own family -> env XML map (the ``_XML_PATH`` of each env class).
FAMILY_XML = {
    "water_plant": "arena_arm_hand_plant.xml",
    "click_mouse": "arena_arm_hand_monitor_mouse.xml",
    "hammer_nail": "arena_arm_hand_hammer_nail.xml",
    "pick_bucket": "arena_arm_hand_bucket_pick.xml",
    "pinch_tongs": "arena_arm_hand_table_tongs.xml",
    "fold_glasses": "arena_arm_hand_glass_v2.xml",
    "bimanual_hanoi": "arena_arm_hand_bimanual_hanoi.xml",
    "bimanual_microwave_cook": "arena_arm_hand_microwave_cook.xml",
    "bimanual_unlock_ipad": "arena_arm_hand_ipad.xml",
    "bimanual_assembly": "arena_arm_hand_bimanual_assembly.xml",
    "bimanual_photograph": "arena_arm_hand_bimanual_photograph.xml",
}

#: MuJoCo joint type codes -> names.
_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def _absolutize_world_xml(path: Path) -> list[str]:
    """Rewrite every relative ``file=`` in a saved world XML to the mirror."""
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []

    def fix(match: re.Match[str]) -> str:
        ref = match.group(1)
        if Path(ref).is_absolute() or Path(path.parent / ref).is_file():
            return match.group(0)
        candidate = XMLS / ref
        if candidate.is_file():
            return match.group(0).replace(ref, candidate.resolve().as_posix())
        # Mesh refs saved relative to the original xmls dir may carry a
        # leading directory that already matches the mirror layout.
        parts = Path(ref).parts
        for i in range(len(parts)):
            tail = Path(*parts[i:]) if i else Path(ref)
            if (XMLS / tail).is_file():
                return match.group(0).replace(ref, (XMLS / tail).resolve().as_posix())
        missing.append(ref)
        return match.group(0)

    text = re.sub(r'file="([^"]+)"', fix, text)
    path.write_text(text, encoding="utf-8")
    return sorted(set(missing))


def extract_one(task: str) -> dict[str, Any]:
    """Replay one instance.npz onto its env XML; write the world package."""
    import mujoco

    npz_path = TASKS / f"{task}-privileged" / "environment" / "instance.npz"
    z = np.load(npz_path, allow_pickle=False)
    family = str(z["task"])
    xml_rel = FAMILY_XML.get(family)
    if xml_rel is None:
        raise SystemExit(f"family {family!r} 无 env XML 映射")
    xml_path = XMLS / xml_rel
    if not xml_path.is_file():
        raise SystemExit(f"env XML 缺失: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # Replay the restorers' model-array edits (in place, with dtype cast).
    applied = 0
    for key in z.files:
        if not key.startswith("model__"):
            continue
        attr = key[len("model__") :]
        if not hasattr(model, attr):
            raise SystemExit(f"{task}: MjModel 无属性 {attr!r}")
        target = getattr(model, attr)
        value = np.asarray(z[key])
        if isinstance(target, np.ndarray):
            if target.shape != value.shape:
                raise SystemExit(
                    f"{task}: {attr} 形状 {target.shape} != npz {value.shape}"
                )
            target[...] = value
        else:
            setattr(model, attr, value.item() if value.ndim == 0 else value)
        applied += 1

    nq, nv, na = int(model.nq), int(model.nv), int(model.na)
    physics = np.asarray(z["physics"], dtype=np.float64).reshape(-1)
    # rcb_dexjoco saved mj_getState(..., mjSTATE_INTEGRATION): in MuJoCo 3.4
    # that bitmask is TIME | QPOS | QVEL | ACT | WARMSTART | CTRL |
    # QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT.
    import mujoco

    parts: list[tuple[str, int]] = []
    cursor = 0
    for name in (
        "mjSTATE_TIME",
        "mjSTATE_QPOS",
        "mjSTATE_QVEL",
        "mjSTATE_ACT",
        "mjSTATE_WARMSTART",
        "mjSTATE_CTRL",
        "mjSTATE_QFRC_APPLIED",
        "mjSTATE_XFRC_APPLIED",
        "mjSTATE_EQ_ACTIVE",
        "mjSTATE_MOCAP_POS",
        "mjSTATE_MOCAP_QUAT",
    ):
        width = mujoco.mj_stateSize(model, getattr(mujoco.mjtState, name))
        parts.append((name, width))
    total = sum(w for _, w in parts)
    if physics.size != total:
        raise SystemExit(
            f"{task}: physics {physics.size} != mjSTATE_INTEGRATION {total}"
        )
    segments: dict[str, np.ndarray] = {}
    cursor = 0
    for name, width in parts:
        segments[name] = physics[cursor : cursor + width]
        cursor += width
    qpos = segments["mjSTATE_QPOS"].copy()
    qvel = segments["mjSTATE_QVEL"].copy()
    act = segments["mjSTATE_ACT"].copy()
    warmstart = segments["mjSTATE_WARMSTART"].copy()
    ctrl = segments["mjSTATE_CTRL"].copy()

    out_dir = WORLDS
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_out = out_dir / f"{task}.xml"
    mujoco.mj_saveLastXML(str(xml_out), model)
    missing = _absolutize_world_xml(xml_out)

    np.save(out_dir / f"{task}.qpos.npy", qpos)
    np.save(out_dir / f"{task}.qvel.npy", qvel)
    np.save(out_dir / f"{task}.warmstart.npy", warmstart)

    attrs = json.loads(str(z["attrs"]))
    joints = []
    for i in range(model.njnt):
        joints.append(
            {
                "name": model.jnt(i).name,
                "type": _JOINT_TYPES.get(
                    int(model.jnt_type[i]), str(int(model.jnt_type[i]))
                ),
                "qposadr": int(model.jnt_qposadr[i]),
                "qveladr": int(model.jnt_dofadr[i]),
                "body": model.body(int(model.jnt_bodyid[i])).name,
            }
        )
    bodies = {}
    body_pos = np.asarray(model.body_pos).copy()
    body_quat = np.asarray(model.body_quat).copy()
    for i in range(1, model.nbody):
        name = model.body(i).name
        bodies[name] = {
            "world": [round(float(v), 6) for v in body_pos[i]],
            "world_quat": [round(float(v), 6) for v in body_quat[i]],
        }

    def _rot(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        w, x, y, z = (float(v) for v in quat)
        u = np.array([x, y, z], dtype=float)
        return vec + 2.0 * np.cross(u, np.cross(u, vec) + w * vec)

    sites = {}
    for i in range(model.nsite):
        name = model.site(i).name
        body_id = int(model.site_bodyid[i])
        bpos = np.asarray(model.body_pos[body_id], dtype=float)
        bquat = np.asarray(model.body_quat[body_id], dtype=float)
        local = np.asarray(model.site_pos[i], dtype=float)
        xpos = bpos + _rot(bquat, local)
        sites[name] = {
            "body": model.body(body_id).name,
            "local": [round(float(v), 6) for v in model.site_pos[i]],
            "local_quat": [round(float(v), 6) for v in model.site_quat[i]],
            "world": [round(float(v), 6) for v in xpos],
        }

    meta = {
        "suite": "dexjoco",
        "task": task,
        "family": family,
        "env_xml": xml_rel,
        "instance_sha": str(z["sha"]),
        "npz_meta": json.loads(str(z["meta"])),
        "nq": nq,
        "nv": nv,
        "na": na,
        "nu": int(model.nu),
        "model_arrays_applied": applied,
        "mocap_pos": np.asarray(z["mocap_pos"]).tolist(),
        "mocap_quat": np.asarray(z["mocap_quat"]).tolist(),
        "ctrl": ctrl.tolist(),
        "warmstart": warmstart.tolist(),
        "act_state": act.tolist(),
        "attrs": attrs,
        "joints": joints,
        "bodies": bodies,
        "sites": sites,
    }
    (out_dir / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    if missing:
        print(f"WARNING {task}: {len(missing)} unresolved asset refs", file=sys.stderr)
        for m in missing[:6]:
            print(f"   {m}", file=sys.stderr)
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
            for p in TASKS.glob("*-privileged/environment/instance.npz")
        )
    else:
        names = [args.task]
    failures = 0
    for name in names:
        try:
            meta = extract_one(name)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {name}: {exc}", file=sys.stderr)
            continue
        print(
            f"{name}: family={meta['family']} nq={meta['nq']}/{meta['nv']}"
            f" nu={meta['nu']} arrays={meta['model_arrays_applied']}"
            f" mocap={len(meta['mocap_pos'])}"
        )
    print(f"done: {len(names) - failures} extracted, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
