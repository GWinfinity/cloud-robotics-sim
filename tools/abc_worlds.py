#!/usr/bin/env python3
"""Extract Genesis-ready world packages from mirrored ABC Sim EFE instances.

Every ABC Sim task ships ``environment/instance.npz`` (the privileged task's
``environment/`` and the standard twins' ``environment/sim/`` are
byte-identical): the fully assembled scene MJCF (``scene_xml`` — every
object, variant, size and pose chosen, complete with the state the EFE
suite re-bakes into the file), the full initial ``qpos0`` at rest, the
scene's randomization record (``randomization`` — the directive and the
per-task parameters the upstream evaluator and runtime read), the success
``rule`` tag, the state's ``sha`` and provenance ``meta``.

Upstream anchors asset lookup by ``chdir``-ing into ``abc_sim/models`` (the
XML's ``meshdir="assets/"`` is resolved by MuJoCo against the process CWD);
this tool instead rewrites every relative ``file="..."`` to this machine's
mirror root so Genesis' ``from_xml_string`` path (no base dir, forward-slash
drive paths) compiles the same world without CWD games.

Outputs, per task, under ``assets/abc-sim/worlds/``:

* ``<task>.xml`` — the world MJCF with asset paths rewritten to the mirror;
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen generalised state
  (split sizes from the XML joint walk: free=7/6, ball=4/3, else 1/1; the
  instance records only ``qpos0`` — "at rest" — so ``qvel`` is zeros);
* ``<task>.meta.json`` — family/task/rule/sha, the randomization record, the
  joint inventory (ranges for the servo clamps), and body/site world poses
  computed by walking the XML kinematic tree (static composed poses, free
  joints overridden from ``qpos0``) — the offline-binding poses the RoboCasa
  packs take from the instance's compiled ``model_body_*`` arrays, which the
  ABC instances do not carry.

No upstream runtime is involved: the frozen scene XML is the spec-level
truth, mirroring the EFE suite's "one frozen instance" definition.

Usage::

    uv run python tools/abc_worlds.py --all
    uv run python tools/abc_worlds.py --task abc-sim-ball-tray-i00
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
UPSTREAM = ROOT / "data" / "efe" / "upstream"
TASKS = UPSTREAM / "tasks" / "abc-sim"
WORLDS = ROOT / "assets" / "abc-sim" / "worlds"

#: MuJoCo resolves an asset ``file`` relative to the process CWD first and
#: only then joins it onto ``meshdir``/``texturedir`` (engine_resource.c).
#: Upstream anchors the CWD at the package's ``models/`` dir, so ABC scenes
#: reference files two ways: ``assets/task_put_relative/...`` (exists
#: directly under ``models/``) and ``i2rt_yam/assets/...`` (needs the
#: ``meshdir="assets/"`` join). The rewrite tries the same two candidates in
#: the same order against this machine's mirror.
_MODELS_ROOT = (ROOT / "assets" / "abc-sim" / "models").resolve()

#: MuJoCo joint types -> (nq contribution, nv contribution)
_JOINT_Q = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}

_ABS_FILE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (vectorized over rows)."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternion q (both (N,4)/(N,3) or single rows)."""
    q = np.atleast_2d(q)
    v = np.atleast_2d(v)
    w, x, y, z = q.T
    u = np.stack((x, y, z), axis=-1)
    s = np.stack((w,) * 3, axis=-1)
    out = v + 2.0 * np.cross(u, np.cross(u, v) + s * v)
    return out


def rewrite_xml(xml: str) -> tuple[str, list[str]]:
    """Rewrite relative asset ``file="..."`` paths to the local mirror.

    Returns the rewritten XML and the distinct rewritten paths missing on
    disk (the caller reports them).
    """

    missing: list[str] = []

    def fix(match: re.Match[str]) -> str:
        path = match.group(1)
        if _ABS_FILE.match(path):
            return match.group(0)
        direct = _MODELS_ROOT / path
        joined = _MODELS_ROOT / "assets" / path
        new = str(direct if direct.is_file() else joined)
        if not Path(new).is_file():
            missing.append(new)
        return match.group(0).replace(path, new)

    rewritten = re.sub(r'file="([^"]+)"', fix, xml)
    return rewritten, sorted(set(missing))


def _default_joint_types(root: ET.Element) -> dict[str, str]:
    """Merged ``joint type`` per default class (child classes override)."""
    merged: dict[str, str] = {}

    def visit(element: ET.Element, inherited: str) -> None:
        current = inherited
        for joint in element.findall("joint"):
            jtype = joint.get("type")
            if jtype:
                current = jtype
        name = element.get("class")
        if name:
            merged[name] = current
        for child in element.findall("default"):
            visit(child, current)

    compiler = root.find("default")
    if compiler is not None:
        for child in compiler.findall("default"):
            visit(child, "hinge")
    return merged


def _joint_spec(root: ET.Element) -> tuple[list[dict], int, int]:
    """Walk the worldbody's body tree: per-joint record plus (nq, nv) totals.

    ``root.iter("joint")`` cannot be used here: ``<default>`` class blocks
    and ``<equality>`` constraints also spell ``<joint .../>`` elements, and
    the ABC scene XMLs carry several of each — they define no DOFs. Joint
    *type* also inherits from ``<default class=...>`` chains (e.g. ABC's
    ``ball_tray_ball`` class sets ``type="free"`` on its joints), so the
    element's own ``type`` attribute is only a fallback.
    """
    defaults = _default_joint_types(root)
    joints: list[dict] = []
    nq = nv = 0
    worldbody = root.find("worldbody")
    bodies = worldbody.iter("body") if worldbody is not None else ()
    for body in bodies:
        for element in body:
            forced_type = ""
            if element.tag == "freejoint":
                forced_type = "free"
            elif element.tag != "joint":
                continue
            jtype = forced_type or element.get("type") or defaults.get(
                element.get("class", ""), "hinge"
            )
            dnq, dnv = _JOINT_Q.get(jtype, (1, 1))
            nq += dnq
            nv += dnv
            joints.append(
                {
                    "name": element.get("name", ""),
                    "type": jtype,
                    "range": element.get("range"),
                    "limited": element.get("limited"),
                    "body": body.get("name", ""),
                }
            )
    return joints, nq, nv


def _world_poses(
    root: ET.Element, joints: list[dict], qpos: np.ndarray
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Body/site world poses from the XML tree, free joints from ``qpos``.

    Body numbering follows XML document order, so walking ``worldbody``
    depth-first and composing ``pos``/``quat`` gives the same poses MuJoCo
    compiles for joint-free chains; a body's leading free/ball joint instead
    takes its translation (and ball rotation) from the ``qpos`` segment.
    Joints inheriting ``type`` from a ``<default class=...>`` chain resolve
    through the same map :func:`_joint_spec` uses.
    """
    defaults = _default_joint_types(root)
    bodies: dict[str, dict] = {}
    sites: dict[str, dict] = {}
    cursor = 0

    def walk(element: ET.Element, parent_pos: np.ndarray, parent_quat: np.ndarray) -> None:
        nonlocal cursor
        for body in element.findall("body"):
            name = body.get("name", "")
            pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
            quat = np.array(
                [float(v) for v in body.get("quat", "1 0 0 0").split()]
            )
            world_pos = parent_pos + _quat_rot(parent_quat, pos)[0]
            world_quat = _quat_mul(parent_quat, quat[None, :])[0]
            for joint in body:
                if joint.tag not in ("joint", "freejoint"):
                    continue
                jtype = "free" if joint.tag == "freejoint" else (
                    joint.get("type")
                    or defaults.get(joint.get("class", ""), "hinge")
                )
                width = _JOINT_Q.get(jtype, (1, 1))[0]
                if jtype == "free" and cursor + 7 <= qpos.size:
                    world_pos = qpos[cursor : cursor + 3].copy()
                    world_quat = qpos[cursor + 3 : cursor + 7].copy()
                elif jtype == "ball" and cursor + 4 <= qpos.size:
                    world_quat = qpos[cursor : cursor + 4].copy()
                cursor += width
            if name:
                bodies[name] = {
                    "world": [round(float(v), 6) for v in world_pos],
                    "world_quat": [round(float(v), 6) for v in world_quat],
                }
                for site in body.findall("site"):
                    sname = site.get("name", "")
                    spos = np.array(
                        [float(v) for v in site.get("pos", "0 0 0").split()]
                    )
                    spos = world_pos + _quat_rot(world_quat, spos)[0]
                    sites[sname] = {
                        "body": name,
                        "world": [round(float(v), 6) for v in spos],
                    }
            walk(body, world_pos, world_quat)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        ident = np.array([1.0, 0.0, 0.0, 0.0])
        origin = np.zeros(3)
        walk(worldbody, origin, ident)
    return bodies, sites


def extract_one(task: str, instance: Path, check_assets: bool) -> dict:
    """Extract one instance.npz into the world package; return its meta."""
    data = np.load(instance, allow_pickle=False)
    family = str(data["family"])
    task_name = str(data["task"])
    xml = str(data["scene_xml"])
    qpos0 = np.asarray(data["qpos0"], dtype=np.float64).reshape(-1)
    randomization = json.loads(str(data["randomization"]))
    rule = str(data["rule"])
    sha = str(data["sha"])
    meta_src = json.loads(str(data["meta"]))

    root = ET.fromstring(xml)
    joints, nq, nv = _joint_spec(root)
    if qpos0.size != nq:
        raise ValueError(f"{task}: qpos0 has {qpos0.size}, XML joint walk gives nq={nq}")
    bodies, sites = _world_poses(root, joints, qpos0)

    rewritten, missing = rewrite_xml(xml)

    out_dir = WORLDS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}.xml").write_text(rewritten, encoding="utf-8")
    np.save(out_dir / f"{task}.qpos.npy", qpos0)
    np.save(out_dir / f"{task}.qvel.npy", np.zeros(nv, dtype=np.float64))

    meta = {
        "suite": "abc-sim",
        "task": task,
        "family": family,
        "upstream_task": task_name,
        "rule": rule,
        "instance_sha": sha,
        "nq": nq,
        "nv": nv,
        "randomization": randomization,
        "upstream_meta": meta_src,
        "joints": joints,
        "bodies": bodies,
        "sites": sites,
    }
    (out_dir / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    if check_assets and missing:
        print(
            f"WARNING {task}: {len(missing)} referenced files missing:", file=sys.stderr
        )
        for path in missing[:8]:
            print(f"   {path}", file=sys.stderr)
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--no-check", action="store_true", help="skip missing-asset warnings"
    )
    args = parser.parse_args(argv)
    if not args.all and not args.task:
        raise SystemExit("give --task <name> or --all")

    if args.all:
        instances = sorted(TASKS.glob("*-privileged/environment/instance.npz"))
    else:
        instances = [TASKS / f"{args.task}-privileged/environment/instance.npz"]
    failures = 0
    for instance in instances:
        task = instance.parent.parent.name.removesuffix("-privileged")
        try:
            meta = extract_one(task, instance, check_assets=not args.no_check)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {task}: {exc}", file=sys.stderr)
            continue
        print(
            f"{task}: family={meta['family']} bodies={len(meta['bodies'])} "
            f"joints={meta['nq']}/{meta['nv']} sites={len(meta['sites'])}"
        )
    print(f"done: {len(instances) - failures} extracted, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
