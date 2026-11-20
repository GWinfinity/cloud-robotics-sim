#!/usr/bin/env python3
"""Compose Genesis-ready HumanoidBench world packages from the EFE instances.

The 10 HumanoidBench tasks (4 locomotion + 6 manipulation, Unitree H1 /
H1+Shadow hands) are rebuilt exactly as the EFE tooling runs them: the
patched upstream package (carlosferrazza/humanoid-bench @ cb118903 with the
EFE image's ``no-torch`` / ``upstream-fixes`` patches and ``box_feet``
applied, probe venv ``.hbvenv`` with mujoco 3.3.7 + dm_control 1.0.34), a
``rcb_hb.Session`` opened on the task's frozen ``instance.npz``, and the
composed dm_control model exported with ``mj_saveLastXML``. Writes under
``assets/humanoidbench/worlds/``:

* ``<task>.xml`` — the flattened world (asset paths rewritten to the local
  mirror roots);
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen t = 0 state;
* ``<task>.meta.json`` — success bar, episode cap, per-task termination
  spec, class/module constants, the frozen task state (vars minus
  reward-shaped fields), site local poses, and the robot link/mass table
  for centre-of-mass velocity.

Usage::

    .hbvenv/Scripts/python.exe tools/hb_worlds.py --all
    .hbvenv/Scripts/python.exe tools/hb_worlds.py --task humanoidbench-loco-walk-i00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "humanoidbench"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "humanoidbench"
WORLDS = ASSETS / "worlds"
EFE_IMAGE = ROOT / "data" / "efe" / "upstream" / "hb-src" / "efe-image"
UPSTREAM_ASSETS = ASSETS / "humanoid_bench" / "assets"

#: Site table shared by the reward ports (local pose recorded per task).
_SITES = (
    "head",
    "imu",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
    "wipe_contact_site_a",
    "wipe_contact_site_b",
    "wipe_contact_site_c",
    "wipe_contact_site_d",
    "wipe_contact_site_e",
)


def _session(task: str) -> Any:
    """Open rcb_hb.Session on the task's frozen privileged instance."""
    sys.path.insert(0, str(EFE_IMAGE))
    try:
        import rcb_hb  # noqa: PLC0415

        return rcb_hb.Session(
            instance=TASKS / f"{task}-privileged" / "environment" / "instance.npz"
        )
    finally:
        sys.path.remove(str(EFE_IMAGE))


def _absolutize_xml(xml: str) -> str:
    """Rewrite every file ref to an absolute mirror path (posix slashes)."""
    import re

    def _resolve(match: Any) -> str:
        attr, ref = match.group(1), match.group(2)
        if not ref or ref.startswith(("http://", "https://", "data:")):
            return match.group(0)
        if "\\" in ref:
            ref = ref.replace("\\", "/")
        if Path(ref).is_file():
            return f'{attr}="{ref}"'
        # refs are written relative to the envs/ dir the scene XML lives in
        rel = ref.lstrip("./")
        for base in (UPSTREAM_ASSETS / "envs", UPSTREAM_ASSETS):
            cand = base / rel
            if cand.is_file():
                return f'{attr}="{cand.as_posix()}"'
        hits = [f for f in UPSTREAM_ASSETS.rglob(Path(rel).name) if f.is_file()]
        if not hits:
            raise SystemExit(f"asset not found in mirror: {ref}")
        return f'{attr}="{hits[0].as_posix()}"'

    return re.sub(
        r'(\sfile(?:right|left|up|down|front|back)?)="([^"]+)"', _resolve, xml
    )


def _site_table(model: Any) -> dict[str, dict[str, Any]]:
    """Local pose of every reward-relevant site (missing names skipped)."""
    table: dict[str, dict[str, Any]] = {}
    for name in _SITES:
        try:
            sid = model.site(name).id
        except KeyError:
            continue
        table[name] = {
            "body": model.body(int(model.site_bodyid[sid])).name,
            "local": [float(v) for v in model.site_pos[sid]],
            "local_quat": [float(v) for v in model.site_quat[sid]],
        }
    return table


def _termination_spec(task: str, src: str) -> dict[str, Any]:
    """Classify the task's get_terminated source into a runtime spec."""
    spec: dict[str, Any] = {"task": task}
    if "pivot_sphere_collision" in src:
        spec.update(
            kind="balance_hard",
            qpos_z_lt=0.8,
            sphere_geom="pivot_sphere_collision",
            board_geom="stand_board_collision",
        )
        return spec
    import re

    m = re.search(r"qpos\[2\]\s*<\s*([0-9.]+)", src)
    spec["qpos_z_lt"] = float(m.group(1)) if m else None
    if "task_index == 5" in src:
        spec["kind"] = "bookshelf"
        spec["min_obj_z_lt"] = 0.5
    elif "left_cube_to_rotate" in src:
        spec["kind"] = "cube"
        spec["min_obj_z_lt"] = 0.6
    elif "window_wiping_tool" in src:
        spec["kind"] = "window"
        spec["tool_z_lt"] = 0.58
    elif spec["qpos_z_lt"] is not None:
        spec["kind"] = "qpos_z"
    else:
        spec["kind"] = "never"
    return spec


def _jsonable(value: Any) -> Any:
    """JSON-safe copy of a task-state value (ndarrays tagged, like rcb_hb)."""

    def _enc(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(type(obj).__name__)

    return json.loads(json.dumps(value, default=_enc))


def _actuator_moments(model: Any, data: Any) -> list[list[list[float]]]:
    """Sparse qfrc=moment@force columns, probed by finite differences.

    ``data.actuator_moment`` is exposed only in its sparse flat layout, so
    each column is probed exactly: qfrc_actuation is affine in ctrl, and
    ``d qfrc / d ctrl_i = moment[:, i] * gainprm[i, 0]``.
    """
    import mujoco

    nu, nv = int(model.nu), int(model.nv)
    base_ctrl = np.asarray(data.ctrl, dtype=np.float64).copy()
    mujoco.mj_forward(model, data)
    q0 = np.asarray(data.qfrc_actuator, dtype=np.float64).copy()
    eps = 1e-6
    records: list[list[list[float]]] = []
    eye = np.zeros(nu, dtype=np.float64)
    for i in range(nu):
        eye[:] = 0.0
        eye[i] = eps
        data.ctrl[:] = base_ctrl + eye
        mujoco.mj_forward(model, data)
        col = (np.asarray(data.qfrc_actuator, dtype=np.float64) - q0) / eps
        gain = float(model.actuator_gainprm[i, 0])
        if abs(gain) > 1e-12:
            col = col / gain
        records.append(
            [[int(dof), float(col[dof])] for dof in range(nv) if abs(col[dof]) > 1e-10]
        )
    data.ctrl[:] = base_ctrl
    mujoco.mj_forward(model, data)
    return records


def compose_one(task: str) -> dict[str, Any]:
    """Compose one HumanoidBench world package; return its meta."""
    import mujoco

    session = _session(task)
    model, data = session.model, session.data

    xml_path = WORLDS / "_tmp.xml"
    WORLDS.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(xml_path), model)
    xml = _absolutize_xml(xml_path.read_text(encoding="utf-8"))
    xml_path.unlink()

    qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(data.qvel, dtype=np.float64).copy()

    import inspect

    cls = type(session.low.task)
    task_vars = {
        k: v
        for k, v in vars(session.low.task).items()
        if k not in ("_env", "robot", "unwrapped") and "reward" not in k.lower()
    }
    if hasattr(cls, "bookshelf_objects"):
        # upstream stores negative body offsets (numpy-style from the end of
        # the xpos table); the runtime addresses links by name
        task_vars["bookshelf_objects"] = [
            model.body(int(bid) if int(bid) >= 0 else model.nbody + int(bid)).name
            for bid in cls.bookshelf_objects
        ]
    task_state = _jsonable(task_vars)

    rewards = json.loads((EFE_IMAGE / "rewards.json").read_text(encoding="utf-8"))
    short = session.task
    module_globals = rewards.get(short, {}).get("module_globals", {})

    robot_root = "pelvis"
    robot_links = []
    root_id = model.body(robot_root).id
    for b in range(model.nbody):
        if model.body_rootid[b] == root_id or b == root_id:
            name = model.body(b).name
            if name:
                robot_links.append({"name": name, "mass": float(model.body_mass[b])})
    actuated = []
    moments = _actuator_moments(model, data)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        adr = int(model.jnt_dofadr[jid])
        actuated.append(
            {
                "name": model.actuator(i).name,
                "joint": model.joint(jid).name,
                "dofadr": adr,
                "ctrlrange": [float(v) for v in model.actuator_ctrlrange[i]],
                # sparse qfrc = moment @ actuator_force row (tendon-routed
                # hands map one actuator to several dofs; the runtime
                # recovers the upstream per-actuator force from the dof
                # forces by lstsq)
                "moment": moments[i],
            }
        )

    terminated_src = inspect.getsource(cls.get_terminated)
    meta: dict[str, Any] = {
        "suite": "humanoidbench",
        "task": task,
        "task_short": short,
        "env_id": session.env_id,
        "robot": "h1hand" if "h1hand" in session.env_id else "h1",
        "success_bar": float(session.success_bar),
        "max_steps": int(session.max_steps),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "timestep": float(model.opt.timestep),
        "frame_skip": 10,
        "sites": _site_table(model),
        "termination": _termination_spec(short, terminated_src),
        "class_consts": {
            k: getattr(cls, k)
            for k in ("_move_speed",)
            if isinstance(getattr(cls, k, None), (int, float))
        },
        "module_globals": module_globals,
        "task_state": task_state,
        "room_objects": (
            ["chair", "trophy", "headphone", "package_a", "package_b", "snow_globe"]
            if short == "room"
            else []
        ),
        "robot_links": robot_links,
        "actuated": actuated,
        "tool_subtree": (
            [
                model.body(b).name
                for b in range(model.nbody)
                if model.body_rootid[b] == model.body("window_wiping_tool").id
                or b == model.body("window_wiping_tool").id
            ]
            if short == "window"
            else []
        ),
        "state_sha": session.state_sha(),
        "upstream": {"humanoid-bench": "cb1189039151c8aadaaa987b442da54383c87fab"},
    }
    (WORLDS / f"{task}.xml").write_text(xml, encoding="utf-8")
    np.save(WORLDS / f"{task}.qpos.npy", qpos)
    np.save(WORLDS / f"{task}.qvel.npy", qvel)
    (WORLDS / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"{task}: env={session.env_id} nq={len(qpos)} nu={model.nu} "
        f"bar={session.success_bar} links={len(robot_links)}"
    )
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
            compose_one(name)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"done: {len(names) - failures} composed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
