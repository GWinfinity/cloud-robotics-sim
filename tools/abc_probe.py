#!/usr/bin/env python3
"""Probe the mirrored ABC Sim instances with upstream MuJoCo code.

Runs in the probe venv (``.abcvenv``, mujoco==3.6.0 + numpy; the mirrored
``abc_sim`` package goes on ``PYTHONPATH``); the main project venv has no
MuJoCo and must not gain one. For every task this tool:

* builds the env exactly like the EFE runtime's ``rcb_abc.build_env`` and
  replays the episode-reset order of ``rcb_abc.Session._episode_reset``
  (clean data, frozen ``qpos0`` at rest, runtime ``after_reset``, evaluator
  ``reset`` + ``configure_from_randomization``, ``mj_forward``);
* recomputes the instance state hash with ``rcb_abc``'s exact algorithm and
  compares it against the ``sha`` recorded in the npz — the bit-exact
  "the mirrored scene XML reproduces the frozen instance" proof;
* evaluates the upstream evaluator once at the frozen state and records the
  metrics (which embed the geometry-derived constants: box interiors, bin
  cones, apothems, object name lists, ...) plus each runtime's
  ``debug_state``;
* dumps a ``model_card``: the 14 action coordinates' joint/actuator facts
  (ranges for the servo clamps, ctrlranges, gains), the finger equality
  couplings, and the model totals the Genesis runtime port needs.

Output: ``data/efe/compile/abc-sim.probe.json`` — the pinned constants the
predicates table and the Genesis ``EfeAbcTask`` read.

Usage::

    .abcvenv/Scripts/python.exe tools/abc_probe.py --all
    .abcvenv/Scripts/python.exe tools/abc_probe.py --task abc-sim-ball-tray-i00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "efe" / "upstream" / "abc-src"
TASKS = ROOT / "data" / "efe" / "upstream" / "tasks" / "abc-sim"
WORLDS = ROOT / "assets" / "abc-sim" / "worlds"
OUT = ROOT / "data" / "efe" / "compile" / "abc-sim.probe.json"

sys.path.insert(0, str(SRC))
os.environ.setdefault("ABC_SIM_ROOT", str(SRC))

ARM_JOINTS = [f"left_joint{i}" for i in range(1, 7)] + [
    f"right_joint{i}" for i in range(1, 7)
]
ACTION_FINGERS = ("left_left_finger", "right_left_finger")


def _jsonable(value):  # rcb_abc._jsonable equivalent
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_MODEL_SKIP = (
    "mesh_",
    "tex_",
    "hfield_",
    "skin_",
    "name_",
    "names",
    "paths",
    "text_",
    "flex_",
    "bvh_",
    "oct_",
    "plugin",
)


def _model_arrays(model):  # rcb_abc._model_arrays equivalent

    out = {}
    for name in dir(model):
        if name.startswith("_") or name.startswith(_MODEL_SKIP):
            continue
        try:
            v = getattr(model, name)
        except Exception:  # noqa: BLE001 - upstream swallows these too
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub" and v.size:
            out[name] = v
    return out


def state_sha(model, data):
    """rcb_abc.Session.state_sha: integration state + sorted model arrays."""
    """rcb_abc.Session.state_sha: integration state + sorted model arrays."""
    import mujoco

    sig = mujoco.mjtState.mjSTATE_INTEGRATION
    s = np.empty(mujoco.mj_stateSize(model, sig))
    mujoco.mj_getState(model, data, s, sig)
    h = hashlib.sha1(s.tobytes())
    for k, v in sorted(_model_arrays(model).items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()[:16]


def load_instance(path: Path) -> dict:
    """Read one EFE ABC instance.npz into plain Python/numpy values."""
    d = np.load(path, allow_pickle=False)
    return {
        "family": str(d["family"]),
        "task": str(d["task"]),
        "scene_xml": str(d["scene_xml"]),
        "qpos0": np.asarray(d["qpos0"], dtype=np.float64),
        "randomization": json.loads(str(d["randomization"])),
        "rule": str(d["rule"]),
        "sha": str(d["sha"]),
        "meta": json.loads(str(d["meta"])),
    }


def build_env(task: str, scene_xml: str):
    """rcb_abc.build_env verbatim (the source tree now carries the assets)."""
    from abc_sim.config import get_i2rt_sim_config
    from abc_sim.env import MuJoCoYAMEnv
    from abc_sim.task_registry import get_task_physics_defaults, resolve_task

    rt = resolve_task(task)
    env_task = rt.env_task or task
    prompt = rt.task_spec.prompt if rt.task_spec is not None else task.replace("_", " ")
    env = MuJoCoYAMEnv(
        config=get_i2rt_sim_config(),
        render_cameras=False,
        prompt=prompt,
        scene_xml=None,
        **get_task_physics_defaults(env_task),
        max_episode_steps=None,
        terminate_on_success=False,
    )
    env.reload_from_xml(scene_xml)
    env.set_task(task)
    env._task_randomizer = None
    return env


def episode_reset(env, inst, rstate) -> None:
    """rcb_abc.Session._episode_reset order."""
    import mujoco

    env.reset(seed=0, randomize=False)
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:] = inst["qpos0"]
    env.data.qvel[:] = 0.0
    env._last_randomization = rstate
    if rstate.metadata.get("prompt") is not None:
        env.prompt = str(rstate.metadata["prompt"])
    if env._task_runtime is not None:
        env._task_runtime.after_reset(env.model, env.data, randomization=rstate)
        env._sync_prompt_from_task_runtime()
    ev = env._task_evaluator
    if ev is not None:
        ev.reset(nworld=1)
        if hasattr(ev, "configure_from_randomization"):
            ev.configure_from_randomization(env.model, rstate)
        for hook, key in (
            ("set_active_trash_joints", "trash_joints"),
            ("set_active_object_joints", "active_object_joints"),
        ):
            if hasattr(ev, hook) and rstate.metadata.get(key) is not None:
                getattr(ev, hook)(rstate.metadata[key])
    env.cur_step = 0
    mujoco.mj_forward(env.model, env.data)


def model_card(model) -> dict:
    """The 14 action coordinates' servo facts + finger equalities."""
    import mujoco

    card: dict = {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}
    # 14-dim action layout: 6 left arm, left finger, 6 right arm, right finger
    ordered = (
        ARM_JOINTS[:6] + ["left_left_finger"] + ARM_JOINTS[6:] + ["right_left_finger"]
    )
    entries = []
    for name in ordered:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        entries.append(
            {
                "joint": name,
                "qposadr": int(model.jnt_qposadr[jid]),
                "qveladr": int(model.jnt_dofadr[jid]),
                "range": [float(v) for v in model.jnt_range[jid]],
                "limited": int(model.jnt_limited[jid]),
                "type": int(model.jnt_type[jid]),
            }
        )
    card["action_qposadr"] = [e["qposadr"] for e in entries]
    card["action_qveladr"] = [e["qveladr"] for e in entries]
    card["action_joints"] = entries
    actuators = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if not name:
            continue
        actuators.append(
            {
                "name": name,
                "joint": model.joint(int(model.actuator_trnid[i][0])).name,
                "ctrlrange": [float(v) for v in model.actuator_ctrlrange[i]],
                "gainprm": [float(v) for v in model.actuator_gainprm[i]],
                "biasprm": [float(v) for v in model.actuator_biasprm[i]],
            }
        )
    card["actuators"] = actuators
    equalities = []
    for i in range(model.neq):
        if int(model.eq_type[i]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        equalities.append(
            {
                "joint1": model.joint(int(model.eq_obj1id[i])).name,
                "joint2": model.joint(int(model.eq_obj2id[i])).name,
                # MuJoCo >= 3.3 keeps the joint-equality polynomial in
                # eq_data's first five slots (eq_polycoef was removed).
                "polycoef": [float(v) for v in model.eq_data[i][:5]],
            }
        )
    card["equalities"] = equalities
    card["timestep"] = float(model.opt.timestep)
    card["integrator"] = int(model.opt.integrator)
    return card


def probe_one(task: str) -> dict:
    """Probe one task: exact-hash proof plus evaluator/runtime/model facts."""
    from abc_sim.randomization.core import RandomizationState

    inst = load_instance(TASKS / f"{task}-privileged" / "environment" / "instance.npz")
    # Full-fidelity build (rcb_abc order): default-scene constructor, then
    # the instance swap, with upstream's own asset anchoring — the recorded
    # sha must reproduce exactly.
    env = build_env(inst["task"], inst["scene_xml"])
    model, data = env.model, env.data
    if inst["qpos0"].shape != (model.nq,):
        raise ValueError(f"{task}: qpos0 {inst['qpos0'].shape} vs nq {model.nq}")
    rstate = RandomizationState.from_dict(inst["randomization"])
    episode_reset(env, inst, rstate)

    sha = state_sha(model, data)

    # World-pack cross-check: the rewritten XML (absolute mirror paths) must
    # compile to the same physics state — the pack is what Genesis spawns.
    worldpack_xml = (WORLDS / f"{task}.xml").read_text(encoding="utf-8")
    env.reload_from_xml(worldpack_xml)
    episode_reset(env, inst, rstate)
    worldpack_sha = state_sha(env.model, env.data)

    result: dict = {
        "family": inst["family"],
        "upstream_task": inst["task"],
        "rule": inst["rule"],
        "instance_sha": inst["sha"],
        "reproduced_sha": sha,
        "instance_exact": sha == inst["sha"],
        "worldpack_sha": worldpack_sha,
        "worldpack_matches": worldpack_sha == sha,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "model_card": model_card(model),
        "randomization": inst["randomization"],
    }

    ev = env._task_evaluator
    if ev is not None:
        res = ev.evaluate_qpos_batch(inst["qpos0"][None, :].astype(np.float32))
        result["nop_success"] = bool(res.success[0])
        result["nop_reward"] = float(res.reward[0])
        result["nop_metrics"] = _jsonable(res.metrics)
        result["evaluator_attrs"] = _scrape_evaluator(ev)
    rt = env._task_runtime
    if rt is not None:
        result["runtime_debug_state"] = _jsonable(rt.debug_state())
    return result


def _scrape_evaluator(ev: Any) -> dict[str, Any]:
    """Collect the resolver-facing constants each evaluator bound at init."""
    out: dict[str, Any] = {}
    names = (
        "gap_min_m",
        "gap_max_m",
        "row_dx_max_m",
        "z_tolerance_m",
        "upright_dot_min",
        "position_tolerance_m",
        "height_tolerance_m",
        "upright_z_min",
        "_table_x_bounds",
        "_table_y_bounds",
        "_table_z",
        "_container_radius",
        "_container_floor_z",
        "_container_rim_z",
        "_target_joint_name",
        "_bin_bottom_y",
        "_bin_top_y",
        "_bin_radius_bottom",
        "_bin_radius_top",
        "_goal_local_center",
        "_goal_half_size",
        "_nuts_goal_local_center",
        "_nuts_goal_size",
        "_bolts_goal_local_center",
        "_bolts_goal_size",
        "_bin_x_bounds",
        "_bin_y_bounds",
        "_bin_floor_z",
        "_bin_rim_z",
        "_axis_xy",
        "_dropped_local_z_threshold_m",
        "_usable_half_extents",
        "floor_tolerance_m",
    )
    for name in names:
        if hasattr(ev, name):
            out[name.lstrip("_")] = _jsonable(getattr(ev, name))
    goal_regions = getattr(ev, "_goal_regions", None)
    if isinstance(goal_regions, dict):
        out["goal_regions"] = {
            str(key): [_jsonable(center), _jsonable(half)]
            for key, (center, half) in goal_regions.items()
        }
    directive = getattr(ev, "_directive", None)
    if isinstance(directive, dict) and "goals_board" in directive:
        out["chess_goals_board"] = _jsonable(directive["goals_board"])
        out["chess_classes"] = _jsonable(directive["classes"])
        out["chess_normal_local"] = _jsonable(directive["normal_local"])
        out["chess_joints"] = _jsonable(directive["joints"])
        if "non_target_goals" in directive:
            out["chess_non_target_goals"] = _jsonable(directive["non_target_goals"])
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if not args.all and not args.task:
        raise SystemExit("give --task <name> or --all")

    if args.all:
        tasks = sorted(
            p.parent.parent.name.removesuffix("-privileged")
            for p in TASKS.glob("*-privileged/environment/instance.npz")
        )
    else:
        tasks = [args.task]

    results: dict = {}
    failures = 0
    for task in tasks:
        try:
            results[task] = probe_one(task)
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            results[task] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"ERROR {task}: {exc}", file=sys.stderr)
        info = results[task]
        print(
            f"{task}: exact={info.get('instance_exact')} "
            f"nop_success={info.get('nop_success')}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.out.relative_to(ROOT)} ({len(results)} tasks, {failures} failed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
